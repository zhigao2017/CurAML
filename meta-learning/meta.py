import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
import  numpy as np

from    learner import Learner
from    copy import deepcopy

max_norm=5

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.args=args
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(self.args, config)


        #'''
        to_optim          = [
                             {'params':self.net.specific_parameters(),'lr':args.backbone_lr}, 
                             {'params':self.net.classifier_parameters(),'lr':args.meta_lr}, 
                             {'params':self.net.specific_ks(),'lr':args.k_lr},  
                             {'params':self.net.shared_lrs(),'lr':args.meta_lr},
                             #{'params':self.net.shared_klrs(),'lr':args.meta_lr},
                             {'params':self.net.shared_classifier_lrs(),'lr':args.meta_lr},
                             {'params':self.net.slf_attn.parameters(),'lr':args.meta_lr},
                             {'params':self.net.controller.parameters(),'lr':args.meta_lr},   
                             {'params':self.net.regularizer.parameters(),'lr':args.meta_lr*0.01},
                             {'params':self.net.optimizer_direction.parameters(),'lr':args.meta_lr*0.01},                                                       
                             ]
        #'''

        '''
        to_optim          = [
                             {'params':self.net.specific_parameters(),'lr':args.backbone_lr}, 
                             {'params':self.net.classifier_parameters(),'lr':args.meta_lr}, 
                             {'params':self.net.specific_ks(),'lr':args.k_lr},  
                             {'params':self.net.shared_lrs(),'lr':args.meta_lr},
                             {'params':self.net.shared_klrs(),'lr':args.meta_lr},
                             {'params':self.net.shared_classifier_lrs(),'lr':args.meta_lr},
                             {'params':self.net.slf_attn.parameters(),'lr':args.meta_lr},
                             {'params':self.net.controller.parameters(),'lr':args.meta_lr},  
                             {'params':self.net.regularizer.parameters(),'lr':args.meta_lr*args.backbone_lr},
                             {'params':self.net.optimizer_direction.parameters(),'lr':args.meta_lr},                                                         
                             ]
        '''


        self.meta_optim = optim.Adam(to_optim, amsgrad=False)






        #self.meta_optim = optim.Adam(self.net.trainable_parameters(), lr=args.meta_lr, amsgrad=False,weight_decay=5e-4)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.meta_optim, T_max=self.args.epoch,
                                                              eta_min=self.args.min_meta_lr)


        total_num = sum(p.numel() for p in self.net.specific_parameters() )
        trainable_num = sum(p.numel() for p in self.net.specific_parameters() if p.requires_grad)


        total_num = sum(p.numel() for p in self.net.shared_parameters() )
        trainable_num = sum(p.numel() for p in self.net.shared_parameters() if p.requires_grad)


        total_num = sum(p.numel() for p in self.net.specific_bns() )
        trainable_num = sum(p.numel() for p in self.net.specific_bns() if p.requires_grad)


        total_num = sum(p.numel() for p in self.net.shared_bns() )
        trainable_num = sum(p.numel() for p in self.net.shared_bns() if p.requires_grad)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)


        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry,step,epoch):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        losses_q_numpy = [0 for _ in range(self.update_step + 1)] 

        loss_meta=0

        for i in range(task_num):

            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector(epoch)

            specific_parameters_fast_weights, classifier_fast_weights, specific_k_fast_weights = self.net.finetune_classifier(x_spt[i], y_spt[i], specific_vars=None, specific_kvars=None, shared_vars=None, bn_training=True)

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], specific_vars=specific_parameters_fast_weights, specific_kvars=specific_k_fast_weights, classifier_vars=classifier_fast_weights, shared_vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])

            specific_parameters_grad = torch.autograd.grad(loss, specific_parameters_fast_weights, retain_graph=True)
            classifier_grad = torch.autograd.grad(loss, classifier_fast_weights , retain_graph=True,allow_unused= True)
            specific_k_grad = torch.autograd.grad(loss, specific_k_fast_weights , allow_unused= True)




            num_parameters=len(specific_parameters_grad)
            num_classifier_parameters=len(classifier_grad)
            num_k=len(specific_k_grad)


            h_0=torch.zeros(2,num_k,1).cuda()
            c_0=torch.zeros(2,num_k,1).cuda()
            hidden=(h_0,c_0)
            input_lstm=torch.zeros(1,num_k,1).cuda()
            for j in range(len(specific_k_grad)):
                input_lstm[0,j,0]=specific_k_grad[j]
            searchdirection, hidden=self.net.optimizer_direction(input_lstm,hidden)
            searchdirection=torch.squeeze(searchdirection)
            k_new_gradient=[]
            for j in range(len(specific_k_grad)):
                k_new_gradient.append(searchdirection[j]+specific_k_grad[j])


            per_step_task_embedding = []
            for v in self.net.specific_ks():
                per_step_task_embedding.append(v.mean())  
            for j in range(len(specific_k_grad)):
                per_step_task_embedding.append(specific_k_grad[j].mean())


            per_step_task_embedding.append((torch.ones(1)*(1)).cuda().mean())

            per_step_task_embedding = torch.stack(per_step_task_embedding)
            per_step_task_embedding = per_step_task_embedding.unsqueeze(0)

            generated_lr= self.net.generate_lr(per_step_task_embedding)
            generated_lr = torch.squeeze(generated_lr)

            k_lr=[]

            for ii in range(num_k):
                k_lr.append(generated_lr[ii])


            num_parameters=len(specific_parameters_grad)
            num_k=len(specific_k_grad)




            specific_parameters_fast_weights = list(map(lambda p: p[1] - self.args.backbone_lr * torch.pow(p[2],2) * p[0], zip(specific_parameters_grad, specific_parameters_fast_weights, self.net.shared_lrs()[0:num_parameters])))
            classifier_fast_weights = list(map(lambda p: p[1] - self.args.update_lr * torch.pow(p[2],2) * p[0], zip(classifier_grad, classifier_fast_weights, self.net.shared_classifier_lrs()[0:num_classifier_parameters])))
            specific_k_fast_weights = list(map(lambda p: p[1] - self.args.k_lr * torch.pow(p[2],2) * p[0], zip(k_new_gradient, specific_k_fast_weights, k_lr)))

           
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], specific_vars=None, specific_kvars=None, classifier_vars=None, shared_vars=None, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q
                #loss_meta=loss_meta+loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct


            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], specific_vars=specific_parameters_fast_weights, specific_kvars=specific_k_fast_weights, classifier_vars=classifier_fast_weights, shared_vars=None, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                loss_meta=loss_meta+per_step_loss_importance_vectors[0]*loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], specific_vars=specific_parameters_fast_weights, specific_kvars=specific_k_fast_weights, classifier_vars=classifier_fast_weights, shared_vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi

                specific_parameters_grad = torch.autograd.grad(loss, specific_parameters_fast_weights, retain_graph=True)
                classifier_grad = torch.autograd.grad(loss, classifier_fast_weights, retain_graph=True,allow_unused= True)
                specific_k_grad = torch.autograd.grad(loss, specific_k_fast_weights, allow_unused= True)




                num_parameters=len(specific_parameters_grad)
                num_classifier_parameters=len(classifier_grad)
                num_k=len(specific_k_grad)


                per_step_task_embedding = []
                for v in specific_k_fast_weights:
                    per_step_task_embedding.append(v.mean())  
                for j in range(len(specific_k_grad)):
                    per_step_task_embedding.append(specific_k_grad[j].mean())


                per_step_task_embedding.append((torch.ones(1)*(k+1)).cuda().mean())

                per_step_task_embedding = torch.stack(per_step_task_embedding)
                per_step_task_embedding = per_step_task_embedding.unsqueeze(0)

                generated_lr= self.net.generate_lr(per_step_task_embedding)
                generated_lr = torch.squeeze(generated_lr)


                k_lr=[]

                for ii in range(num_k):
                    k_lr.append(generated_lr[ii])


                input_lstm=torch.zeros(1,num_k,1).cuda()
                for j in range(len(specific_k_grad)):
                    input_lstm[0,j,0]=specific_k_grad[j]
                searchdirection, hidden=self.net.optimizer_direction(input_lstm,hidden)
                searchdirection=torch.squeeze(searchdirection)
                k_new_gradient=[]
                for j in range(len(specific_k_grad)):
                    k_new_gradient.append(searchdirection[j]+specific_k_grad[j])


                specific_parameters_fast_weights = list(map(lambda p: p[1] - self.args.backbone_lr * torch.pow(p[2],2) * p[0], zip(specific_parameters_grad, specific_parameters_fast_weights, self.net.shared_lrs()[k*num_parameters:(k+1)*num_parameters] )))
                classifier_fast_weights = list(map(lambda p: p[1] - self.args.update_lr * torch.pow(p[2],2) * p[0], zip(classifier_grad, classifier_fast_weights, self.net.shared_classifier_lrs()[k*num_classifier_parameters:(k+1)*num_classifier_parameters])))
                specific_k_fast_weights = list(map(lambda p: p[1] - self.args.k_lr * torch.pow(p[2],2)* p[0], zip(k_new_gradient,specific_k_fast_weights, k_lr)))

                logits_q = self.net(x_qry[i], specific_vars=specific_parameters_fast_weights, specific_kvars=specific_k_fast_weights, classifier_vars=classifier_fast_weights, shared_vars=None, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                loss_meta=loss_meta+per_step_loss_importance_vectors[k]*loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        #loss_meta = loss_meta/(task_num*(self.update_step+1))

        for i in range(self.update_step + 1):
            losses_q_numpy[i]=losses_q[i].cpu().detach().numpy().tolist()

        # optimize theta parameters
        self.meta_optim.zero_grad()
        #loss_q.backward()
        loss_meta.backward()
        self.meta_optim.step()

        self.scheduler.step(epoch+step/500)


        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def get_per_step_loss_importance_vector(self,current_epoch):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.update_step)) * (
                1.0 / self.args.update_step)
        decay_rate = 1.0 / self.args.update_step / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.update_step
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (current_epoch * (self.args.update_step - 1) * decay_rate),
            1.0 - ((self.args.update_step - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).cuda()
        return loss_weights


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        losses_q_numpy = [0 for _ in range(self.update_step_test + 1)] 

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)


        specific_parameters_fast_weights, classifier_fast_weights, specific_k_fast_weights = net.finetune_classifier(x_spt, y_spt,specific_vars=None, specific_kvars=None, shared_vars=None, bn_training=True)

        # 1. run the i-th task and compute loss for k=0

        logits = net(x_spt, specific_vars=specific_parameters_fast_weights, specific_kvars=specific_k_fast_weights, classifier_vars=classifier_fast_weights, shared_vars=None, bn_training=True)
        loss = F.cross_entropy(logits, y_spt)

        specific_parameters_grad = torch.autograd.grad(loss, specific_parameters_fast_weights, retain_graph=True)
        classifier_grad = torch.autograd.grad(loss, classifier_fast_weights, retain_graph=True,allow_unused= True)
        specific_k_grad = torch.autograd.grad(loss, specific_k_fast_weights,allow_unused= True)

        num_parameters=len(specific_parameters_grad)
        num_classifier_parameters=len(classifier_grad)
        num_k=len(specific_k_grad)


        per_step_task_embedding = []
        for v in net.specific_ks():
            per_step_task_embedding.append(v.mean())  
        for j in range(len(specific_k_grad)):
            per_step_task_embedding.append(specific_k_grad[j].mean())


        per_step_task_embedding.append((torch.ones(1)*(1)).cuda().mean())

        per_step_task_embedding = torch.stack(per_step_task_embedding)
        per_step_task_embedding = per_step_task_embedding.unsqueeze(0)

        generated_lr= net.generate_lr(per_step_task_embedding)
        generated_lr = torch.squeeze(generated_lr)

        k_lr=[]

        for ii in range(num_k):
            k_lr.append(generated_lr[ii])



        h_0=torch.zeros(2,num_k,1).cuda()
        c_0=torch.zeros(2,num_k,1).cuda()
        hidden=(h_0,c_0)
        input_lstm=torch.zeros(1,num_k,1).cuda()
        for j in range(len(specific_k_grad)):
            input_lstm[0,j,0]=specific_k_grad[j]
        searchdirection, hidden=net.optimizer_direction(input_lstm,hidden)
        searchdirection=torch.squeeze(searchdirection)
        k_new_gradient=[]
        for j in range(len(specific_k_grad)):
            k_new_gradient.append(searchdirection[j]+specific_k_grad[j])


        specific_parameters_fast_weights = list(map(lambda p: p[1] - self.args.backbone_lr * torch.pow(p[2],2) * p[0], zip(specific_parameters_grad, specific_parameters_fast_weights, net.shared_lrs()[0:num_parameters])))
        classifier_fast_weights = list(map(lambda p: p[1] - self.args.update_lr * torch.pow(p[2],2) * p[0], zip(classifier_grad, classifier_fast_weights, net.shared_classifier_lrs()[0:num_classifier_parameters] )))
        specific_k_fast_weights = list(map(lambda p: p[1] - self.args.k_lr * torch.pow(p[2],2) * p[0], zip(k_new_gradient, specific_k_fast_weights, k_lr)))
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, specific_vars=None, specific_kvars=None, classifier_vars=None, shared_vars=None, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q_numpy[0]=losses_q_numpy[0]+loss_q.detach()
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct


        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, specific_vars=specific_parameters_fast_weights, specific_kvars=specific_k_fast_weights, classifier_vars=classifier_fast_weights, shared_vars=None, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q_numpy[1]=losses_q_numpy[1]+loss_q.detach()            
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, specific_vars=specific_parameters_fast_weights, specific_kvars=specific_k_fast_weights, classifier_vars=classifier_fast_weights, shared_vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            specific_parameters_grad = torch.autograd.grad(loss, specific_parameters_fast_weights,retain_graph=True,allow_unused=True)
            classifier_grad = torch.autograd.grad(loss, classifier_fast_weights, retain_graph=True,allow_unused= True)
            specific_k_grad = torch.autograd.grad(loss, specific_k_fast_weights, allow_unused= True)



            num_parameters=len(specific_parameters_grad)
            num_classifier_parameters=len(classifier_grad)
            num_k=len(specific_k_grad)


            per_step_task_embedding = []
            for v in specific_k_fast_weights:
                per_step_task_embedding.append(v.mean())  
            for j in range(len(specific_k_grad)):
                per_step_task_embedding.append(specific_k_grad[j].mean())


            per_step_task_embedding.append((torch.ones(1)*(k+1)).cuda().mean())

            per_step_task_embedding = torch.stack(per_step_task_embedding)
            per_step_task_embedding = per_step_task_embedding.unsqueeze(0)

            generated_lr= net.generate_lr(per_step_task_embedding)
            generated_lr = torch.squeeze(generated_lr)

            k_lr=[]

            for ii in range(num_k):
                k_lr.append(generated_lr[ii])



            input_lstm=torch.zeros(1,num_k,1).cuda()
            for j in range(len(specific_k_grad)):
                input_lstm[0,j,0]=specific_k_grad[j]
            searchdirection, hidden=net.optimizer_direction(input_lstm,hidden)
            searchdirection=torch.squeeze(searchdirection)
            k_new_gradient=[]
            for j in range(len(specific_k_grad)):
                k_new_gradient.append(searchdirection[j]+specific_k_grad[j])

                
            if k<4:
                specific_parameters_fast_weights = list(map(lambda p: p[1] - self.args.backbone_lr * torch.pow(p[2],2) * p[0], zip(specific_parameters_grad, specific_parameters_fast_weights, net.shared_lrs()[k*num_parameters:(k+1)*num_parameters] )))
                classifier_fast_weights = list(map(lambda p: p[1] - self.args.update_lr * torch.pow(p[2],2) * p[0], zip(classifier_grad, classifier_fast_weights, net.shared_classifier_lrs()[k*num_classifier_parameters:(k+1)*num_classifier_parameters])))
            else:
                specific_parameters_fast_weights = list(map(lambda p: p[1] - self.args.backbone_lr * torch.pow(p[2],2) * p[0], zip(specific_parameters_grad, specific_parameters_fast_weights, net.shared_lrs()[3*num_parameters:(3+1)*num_parameters] )))
                classifier_fast_weights = list(map(lambda p: p[1] - self.args.update_lr * torch.pow(p[2],2) * p[0], zip(classifier_grad, classifier_fast_weights, net.shared_classifier_lrs()[3*num_classifier_parameters:(3+1)*num_classifier_parameters])))                


            specific_k_fast_weights = list(map(lambda p: p[1] - self.args.k_lr * torch.pow(p[2],2) * p[0], zip(k_new_gradient, specific_k_fast_weights, k_lr)))



            logits_q = net(x_qry, specific_vars=specific_parameters_fast_weights, specific_kvars=specific_k_fast_weights, classifier_vars=classifier_fast_weights, shared_vars=None, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q_numpy[k+1]=losses_q_numpy[k+1]+loss_q.detach()
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        for i in range(self.update_step_test + 1):
            losses_q_numpy[i]=losses_q_numpy[i].cpu().detach().numpy().tolist()

        del net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
