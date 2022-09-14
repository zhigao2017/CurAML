import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np

from hyptorch.pmath import dist_matrix, expmap0, logmap0, RiemannianBatchNorm, expmap, logmap,lambda_x
from hyptorch.nn import ToPoincare

from controller import Controller

def Euclidean_distance(a,b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    dis = 2*torch.sqrt(((a - b)**2).sum(dim=2))
    return dis



def Euclidean_distance_save(a,b):

    n = a.shape[0]
    m = b.shape[0]
    d=a.shape[1]

    one_a=torch.ones(d,m).cuda()
    one_b=torch.ones(d,n).cuda()

    aa=torch.mm(a*a,one_a)
    bb=torch.mm(b*b,one_b)
    ab=torch.mm(a,b.t())

    dis=aa+bb.t()-2*ab
    dis = 2*torch.sqrt(dis)

    return dis


class Learner(nn.Module):
    """

    """

    def __init__(self, args, config):

        super(Learner, self).__init__()

        self.config = config
        self.args = args

        # this dict contains all tensors needed to be optimized

        self.specific_parameter = nn.ParameterList()
        self.shared_parameter = nn.ParameterList()

        self.specific_k = nn.ParameterList()
        self.shared_k = nn.ParameterList()

        self.specific_bn = nn.ParameterList()
        self.shared_bn = nn.ParameterList()

        self.classifier_generator = nn.ParameterList() 

        self.shared_lr = nn.ParameterList()
        self.shared_klr = nn.ParameterList()  
        self.classifier_lr = nn.ParameterList()  

        self.classifier = nn.ParameterList()   
        pre_train_model=torch.load('../pretrain/save/classifier_mini-imagenet_resnet12-widecurvautre-1_lr0.05_component16_resnet12projection/epoch-55.pth')

        pre_train_model=pre_train_model['model_sd']

        for j, (name, param) in enumerate(self.config):

            if name is 'block':
                #(input_channel, output_channel, component_num, specific_num, Riemannian_component_num, max_pool_padding, No.layer )
                input_channel=param[0]
                output_channel=param[1]

                component_num=param[2]
                specific_num=param[3]

                shared_num=component_num-specific_num

                input_component_channel=int(input_channel/component_num)
                component_channel=int(output_channel/component_num)

                Riemannian_component_num=param[4]

                layer=param[6] 


                ####conv####
                ##(input_channel, output_channel, component_num, specific_num, Riemannian_component_num, max_pool_padding, No.layer )


                #block  3conv, 3bn,
                w = nn.Parameter(pre_train_model['encoder.layer'+layer+'.conv1.weight'])
                self.specific_parameter.append(w)

                w = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn1.weight'])
                self.specific_parameter.append(w)
                bias= nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn1.bias'])
                self.specific_parameter.append(bias)
                running_mean = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn1.running_mean'], requires_grad=False)
                running_var = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn1.running_var'], requires_grad=False)
                self.specific_bn.extend([running_mean, running_var])


                w = nn.Parameter(pre_train_model['encoder.layer'+layer+'.conv2.weight'])
                self.specific_parameter.append(w)

                w = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn2.weight'])
                self.specific_parameter.append(w)
                bias= nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn2.bias'])
                self.specific_parameter.append(bias)
                running_mean = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn2.running_mean'], requires_grad=False)
                running_var = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn2.running_var'], requires_grad=False)
                self.specific_bn.extend([running_mean, running_var])


                w = nn.Parameter(pre_train_model['encoder.layer'+layer+'.conv3.weight'])
                self.specific_parameter.append(w)

                w = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn3.weight'])
                self.specific_parameter.append(w)
                bias= nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn3.bias'])
                self.specific_parameter.append(bias)
                running_mean = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn3.running_mean'], requires_grad=False)
                running_var = nn.Parameter(pre_train_model['encoder.layer'+layer+'.bn3.running_var'], requires_grad=False)
                self.specific_bn.extend([running_mean, running_var])


                #downsample
                w = nn.Parameter(pre_train_model['encoder.layer'+layer+'.downsample.0.weight'])
                self.specific_parameter.append(w)

                w = nn.Parameter(pre_train_model['encoder.layer'+layer+'.downsample.1.weight'])
                self.specific_parameter.append(w)
                bias = nn.Parameter(pre_train_model['encoder.layer'+layer+'.downsample.1.bias'])
                self.specific_parameter.append(bias)
                running_mean = nn.Parameter(pre_train_model['encoder.layer'+layer+'.downsample.1.running_mean'], requires_grad=False)
                running_var = nn.Parameter(pre_train_model['encoder.layer'+layer+'.downsample.1.running_var'], requires_grad=False)
                self.specific_bn.extend([running_mean, running_var])


                for i in range(specific_num):

                    k = nn.Parameter(pre_train_model['encoder.curvature_'+layer][i] )
                    self.specific_k.append(k)                    
                
                    exp_location = nn.Parameter(pre_train_model['encoder.layer'+layer+'.exp.location'][i,:].unsqueeze(0))
                    self.specific_parameter.append(exp_location)

                    if layer != '4':
                        log_location = nn.Parameter(pre_train_model['encoder.layer'+layer+'.log.location'][i,:].unsqueeze(0))
                        self.specific_parameter.append(log_location)


            elif name is 'distance':
                # ('distance',w,h,compoent_num,all_channel,class_num)
                component_channel=int(param[3]/param[2])
                component_num = param[2] 
                Riemannian_component_num=param[5]
                all_channel=param[3]
                #prototype (component_num,class_num,component_channel)
                #self.prototype = nn.Parameter(torch.randn(param[2],param[4],component_channel))

                #ignore1=nn.Linear(param[3],param[4])
                #w1 = nn.Parameter(ignore1.weight.data)
                #bias1=nn.Parameter(torch.zeros(param[4]))

                w1=nn.Parameter(torch.zeros(param[4],param[3]))
                bias1=nn.Parameter(torch.zeros(param[4]))

                torch.nn.init.kaiming_normal_(w1)
                self.classifier.append(w1)
                self.classifier.append(bias1)

                att=nn.Linear(component_num,component_num)
                att_w=nn.Parameter(att.weight.data)
                att_b=nn.Parameter(torch.zeros(component_num))
                self.classifier.append(att_w)
                self.classifier.append(att_b)   
                

        for j in range (4*len(self.specific_parameter)):
            update_lr=nn.Parameter(1*torch.ones(1))
            self.shared_lr.append(update_lr)

        for j in range (4*len(self.specific_k)):
            k_lr=nn.Parameter(1*torch.ones(1))
            self.shared_klr.append(k_lr)

        for j in range (4*len(self.classifier)):
            classifierlr=nn.Parameter(1*torch.ones(1))
            self.classifier_lr.append(classifierlr)

        num_layers=(len(self.specific_parameter)+len(self.classifier)+len(self.specific_k))*2
        self.slf_attn = nn.Sequential(
            nn.Linear(num_layers, num_layers),
            nn.ReLU(inplace=True),
            nn.Linear(num_layers, num_layers),
            nn.ReLU(inplace=True),                       
            nn.Linear(num_layers, int(num_layers/2) ) ).cuda()
        self.controller = Controller( all_channel, 64,64,5, len(self.specific_k))

        num_layers=(len(self.specific_k))*2
        self.regularizer = nn.Sequential(
            nn.Linear(num_layers+1, num_layers),
            nn.ReLU(inplace=True),
            nn.Linear(num_layers, num_layers),
            nn.ReLU(inplace=True),            
            nn.Linear(num_layers, int(num_layers/2) ) ).cuda()    
                
        checkpoint1 = torch.load('multimanifold_optimizer_intialization_onlyforcurvature/save_n+1dimension_3elayer_sigmoid1e-3-1e5_onlyforcurvature/updatelr_1_backbonelr_1_backbonelr_1_epoch_53_step_499.pth')
        self.regularizer.load_state_dict(checkpoint1)

        self.optimizer_direction = nn.LSTM(1,1,2)
        checkpoint2 = torch.load('multimanifold_optimizer_intialization_searchdirection/lstm2layer_metalr1e-3/updatelr_1_backbonelr_1_backbonelr_1_epoch_11_step_499.pth')
        self.optimizer_direction.load_state_dict(checkpoint2)


    def generate_lr(self,per_step_task_embedding):


        x=self.regularizer(per_step_task_embedding)
        x=F.sigmoid(x*0.001) 
        x=x*10000
        return x



    def generate_l2f(self,per_step_task_embedding):


        x=self.slf_attn(per_step_task_embedding)
        x=F.sigmoid(x) # the valuable domain is [-10,10]
        return x




    def forward(self, input_data, specific_vars=None, specific_kvars=None, classifier_vars=None, shared_vars=None, bn_training=False):


        if specific_vars is None:
            specific_parameter=self.specific_parameter 
        else:
            specific_parameter=specific_vars
           
        if shared_vars is None:
            shared_parameter=self.shared_parameter
            shared_bn=self.shared_bn 
            specific_bn=self.specific_bn 
        else:
            shared_parameter=shared_vars[0]
            shared_bn=shared_vars[1]
            specific_bn=shared_vars[2]

        if specific_kvars ==None:
            specific_k=self.specific_k
        else:
            specific_k=specific_kvars

        if classifier_vars ==  None:
            classifier=self.classifier
        else:
            classifier=classifier_vars
            
        shared_k=self.shared_k

        p_specific_idx = 0
        p_shared_idx = 0
        k_specific_idx = 0
        k_shared_idx = 0
        bn_specific_idx = 0
        bn_shared_idx=0

        for j, (name, param) in enumerate(self.config):
            torch.cuda.empty_cache()
            k_shared_idx=0
            if name is 'block':
                ##(input_channel, output_channel, component_num, specific_num, Riemannian_component_num, max_pool_padding, No.layer )
                component_num=param[2]
                specific_num=param[3]
                shared_num=component_num-specific_num
                in_component_channel=int(param[0]/component_num)
                out_component_channel=int(param[1]/component_num)
                Riemannian_component_num=param[4]
                layer = param[6]
                ####conv####
                ##(input_channel, output_channel, component_num, specific_num, Riemannian_component_num, max_pool_padding, No.layer )
                if layer=='1':
                    
                    f=input_data
                else:
                    f=x # (n,c,h,w)
                ## 3conv
                #1
                w = specific_parameter[p_specific_idx]
                input_data_i=F.conv2d(f, w, bias=None, stride=1, padding=1)
                p_specific_idx += 1

                w_b, b_b = specific_parameter[p_specific_idx], specific_parameter[p_specific_idx + 1]
                running_mean, running_var = specific_bn[bn_specific_idx], specific_bn[bn_specific_idx+1]
                input_data_i = F.batch_norm(input_data_i, running_mean, running_var, weight=w_b, bias=b_b, training=bn_training)
                p_specific_idx += 2
                bn_specific_idx += 2

                input_data_i = F.leaky_relu(input_data_i, negative_slope=0.1, inplace=True) 

                #2
                w = specific_parameter[p_specific_idx]
                input_data_i=F.conv2d(input_data_i, w, bias=None, stride=1, padding=1)
                p_specific_idx += 1

                w_b, b_b = specific_parameter[p_specific_idx], specific_parameter[p_specific_idx + 1]
                running_mean, running_var = specific_bn[bn_specific_idx], specific_bn[bn_specific_idx+1]
                input_data_i = F.batch_norm(input_data_i, running_mean, running_var, weight=w_b, bias=b_b, training=bn_training)
                p_specific_idx += 2
                bn_specific_idx += 2

                input_data_i = F.leaky_relu(input_data_i, negative_slope=0.1, inplace=True) 

                #3
                w = specific_parameter[p_specific_idx]
                input_data_i=F.conv2d(input_data_i, w, bias=None, stride=1, padding=1)
                p_specific_idx += 1

                w_b, b_b = specific_parameter[p_specific_idx], specific_parameter[p_specific_idx + 1]
                running_mean, running_var = specific_bn[bn_specific_idx], specific_bn[bn_specific_idx+1]
                input_data_i = F.batch_norm(input_data_i, running_mean, running_var, weight=w_b, bias=b_b, training=bn_training)
                p_specific_idx += 2
                bn_specific_idx += 2

                #downsample
                w = specific_parameter[p_specific_idx]
                input_data_downsample=F.conv2d(f, w, bias=None, stride=1, padding=0)
                p_specific_idx += 1

                w_b, b_b = specific_parameter[p_specific_idx], specific_parameter[p_specific_idx + 1]
                running_mean, running_var = specific_bn[bn_specific_idx], specific_bn[bn_specific_idx+1]
                input_data_downsample = F.batch_norm(input_data_downsample, running_mean, running_var, weight=w_b, bias=b_b, training=bn_training)
                p_specific_idx += 2
                bn_specific_idx += 2

                input_data_i=input_data_i+input_data_downsample
                input_data_i = F.leaky_relu(input_data_i, negative_slope=0.1, inplace=True) 
                input_data_i=F.max_pool2d(input_data_i, kernel_size=(2,2), stride=2, padding=param[5])


                x=[]
                for i in range(Riemannian_component_num):

                    c_i=specific_k[k_specific_idx]
                    k_specific_idx=k_specific_idx+1
                    exp_location = specific_parameter[p_specific_idx]
                    p_specific_idx=p_specific_idx+1
                    exp_location=expmap0(exp_location,k=c_i)
                    if layer != '4':
                        log_location = specific_parameter[p_specific_idx]
                        p_specific_idx=p_specific_idx+1                            
                        log_location=expmap0(log_location,k=c_i)

                    x_i=torch.index_select(input_data_i,1,torch.arange(i*out_component_channel,(i+1)*out_component_channel).to(input_data_i.device))
                    new_batch, new_channel, new_h, new_w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
                    x_i=torch.transpose(x_i,1,3).contiguous().view(new_batch*new_w*new_h,new_channel)

                    if c_i>0:
                        x_i=x_i-exp_location*torch.sum(x_i*exp_location)
                    else:
                        x_i=(1/(lambda_x(exp_location,c_i)*lambda_x(exp_location,c_i)))*x_i


                    x_i=expmap(exp_location,x_i,k=c_i).squeeze()

                    if layer != '4':
                        x_i=logmap(log_location,x_i,k=c_i)
                    else:
                        x_i=logmap0(x_i,k=c_i)

                    x.append ( torch.transpose(x_i.contiguous().view(new_batch,new_w,new_h,new_channel),1,3) )# (batch,channel,h,w)

                x = torch.cat(x,dim=1) # (n,c,h,w)  



            if name is 'distance':
                # ('distance',w,h,compoent_num,all_channel,class_num)

                component_channel=int(param[3]/param[2])
                feature_num=param[0]*param[1]
                compoent_num=param[2]
                batch=x[0].shape[0]
                channel=x[0].shape[1]
                class_num=param[4]
                Riemannian_component_num=param[5]
                
                f=x
                f = F.adaptive_avg_pool2d(f, (1,1))
                f = f.view(f.shape[0], -1)
                w1, b1 = classifier[0], classifier[1]
                
                #logit=F.linear(f, w1, b1)

                f_expand=f.repeat(1,class_num)
                f_expand=f_expand.reshape(f.shape[0],class_num,param[3])
                w1_expand=w1.repeat(f.shape[0],1,1)

                logit=w1_expand*f_expand
                logit=logit.reshape(f.shape[0],class_num,compoent_num,component_channel)
                logit=torch.sum(logit,3)
                logit=logit.reshape(f.shape[0]*class_num,compoent_num)

                att_w1, att_b1= classifier[2], classifier[3]
                logit_att=F.linear(logit, att_w1, att_b1)
                logit_att=F.softmax(logit_att)
                logit=logit*logit_att*class_num  #(f.shape[0]*class_num,compoent_num)
                logit=torch.sum(logit,1).reshape(f.shape[0],class_num)
                logit=logit+b1

        return  logit




    def finetune_classifier(self, input_data, label, specific_vars=None, specific_kvars=None, classifier_vars=None, shared_vars=None, bn_training=False):

        if specific_vars is None:
            specific_parameter=self.specific_parameter 
        else:
            specific_parameter=specific_vars
           
        if shared_vars is None:
            shared_parameter=self.shared_parameter
            shared_bn=self.shared_bn 
            specific_bn=self.specific_bn 
        else:
            shared_parameter=shared_vars[0]
            shared_bn=shared_vars[1]
            specific_bn=shared_vars[2]

        if specific_kvars ==None:
            specific_k=self.specific_k
        else:
            specific_k=specific_kvars

        if classifier_vars ==  None:
            classifier=self.classifier
        else:
            classifier=classifier_vars
            
        shared_k=self.shared_k

        p_specific_idx = 0
        p_shared_idx = 0
        k_specific_idx = 0
        k_shared_idx = 0
        bn_specific_idx = 0
        bn_shared_idx=0

        for j, (name, param) in enumerate(self.config):
            torch.cuda.empty_cache()
            k_shared_idx=0
            if name is 'block':
                ##(input_channel, output_channel, component_num, specific_num, Riemannian_component_num, max_pool_padding, No.layer )
                component_num=param[2]
                specific_num=param[3]
                shared_num=component_num-specific_num
                in_component_channel=int(param[0]/component_num)
                out_component_channel=int(param[1]/component_num)
                Riemannian_component_num=param[4]
                layer = param[6]
                ####conv####
                ##(input_channel, output_channel, component_num, specific_num, Riemannian_component_num, max_pool_padding, No.layer )
                if layer=='1':
                    
                    f=input_data
                else:
                    f=x # (n,c,h,w)
                ## 3conv
                #1
                w = specific_parameter[p_specific_idx]
                input_data_i=F.conv2d(f, w, bias=None, stride=1, padding=1)
                p_specific_idx += 1

                w_b, b_b = specific_parameter[p_specific_idx], specific_parameter[p_specific_idx + 1]
                running_mean, running_var = specific_bn[bn_specific_idx], specific_bn[bn_specific_idx+1]
                input_data_i = F.batch_norm(input_data_i, running_mean, running_var, weight=w_b, bias=b_b, training=bn_training)
                p_specific_idx += 2
                bn_specific_idx += 2

                input_data_i = F.leaky_relu(input_data_i, negative_slope=0.1, inplace=True) 

                #2
                w = specific_parameter[p_specific_idx]
                input_data_i=F.conv2d(input_data_i, w, bias=None, stride=1, padding=1)
                p_specific_idx += 1

                w_b, b_b = specific_parameter[p_specific_idx], specific_parameter[p_specific_idx + 1]
                running_mean, running_var = specific_bn[bn_specific_idx], specific_bn[bn_specific_idx+1]
                input_data_i = F.batch_norm(input_data_i, running_mean, running_var, weight=w_b, bias=b_b, training=bn_training)
                p_specific_idx += 2
                bn_specific_idx += 2

                input_data_i = F.leaky_relu(input_data_i, negative_slope=0.1, inplace=True) 

                #3
                w = specific_parameter[p_specific_idx]
                input_data_i=F.conv2d(input_data_i, w, bias=None, stride=1, padding=1)
                p_specific_idx += 1

                w_b, b_b = specific_parameter[p_specific_idx], specific_parameter[p_specific_idx + 1]
                running_mean, running_var = specific_bn[bn_specific_idx], specific_bn[bn_specific_idx+1]
                input_data_i = F.batch_norm(input_data_i, running_mean, running_var, weight=w_b, bias=b_b, training=bn_training)
                p_specific_idx += 2
                bn_specific_idx += 2

                #downsample
                w = specific_parameter[p_specific_idx]
                input_data_downsample=F.conv2d(f, w, bias=None, stride=1, padding=0)
                p_specific_idx += 1

                w_b, b_b = specific_parameter[p_specific_idx], specific_parameter[p_specific_idx + 1]
                running_mean, running_var = specific_bn[bn_specific_idx], specific_bn[bn_specific_idx+1]
                input_data_downsample = F.batch_norm(input_data_downsample, running_mean, running_var, weight=w_b, bias=b_b, training=bn_training)
                p_specific_idx += 2
                bn_specific_idx += 2

                input_data_i=input_data_i+input_data_downsample
                input_data_i = F.leaky_relu(input_data_i, negative_slope=0.1, inplace=True) 
                input_data_i=F.max_pool2d(input_data_i, kernel_size=(2,2), stride=2, padding=param[5])


                x=[]
                for i in range(Riemannian_component_num):

                    c_i=specific_k[k_specific_idx]
                    k_specific_idx=k_specific_idx+1
                    exp_location = specific_parameter[p_specific_idx]
                    p_specific_idx=p_specific_idx+1                            
                    exp_location=expmap0(exp_location,k=c_i)
                    if layer != '4':
                        log_location = specific_parameter[p_specific_idx]
                        p_specific_idx=p_specific_idx+1                            
                        log_location=expmap0(log_location,k=c_i)

                    x_i=torch.index_select(input_data_i,1,torch.arange(i*out_component_channel,(i+1)*out_component_channel).to(input_data_i.device))
                    new_batch, new_channel, new_h, new_w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
                    x_i=torch.transpose(x_i,1,3).contiguous().view(new_batch*new_w*new_h,new_channel)

                    if c_i>0:
                        x_i=x_i-exp_location*torch.sum(x_i*exp_location)
                    else:
                        x_i=(1/(lambda_x(exp_location,c_i)*lambda_x(exp_location,c_i)))*x_i

                    x_i=expmap(exp_location,x_i,k=c_i).squeeze()

                    if layer != '4':
                        x_i=logmap(log_location,x_i,k=c_i)
                    else:
                        x_i=logmap0(x_i,k=c_i)

                    x.append ( torch.transpose(x_i.contiguous().view(new_batch,new_w,new_h,new_channel),1,3) )# (batch,channel,h,w)

                x = torch.cat(x,dim=1) # (n,c,h,w)  



            if name is 'distance':
                # ('distance',w,h,compoent_num,all_channel,class_num)

                component_channel=int(param[3]/param[2])
                feature_num=param[0]*param[1]
                compoent_num=param[2]
                batch=x[0].shape[0]
                channel=x[0].shape[1]
                class_num=param[4]
                Riemannian_component_num=param[5]
                
                f=[]

                for i in range (compoent_num):
                    x_i=torch.index_select(x,1,torch.arange(i*component_channel,(i+1)*component_channel).to(x.device))
                    #c_i=specific_k[k_specific_idx]
                    #k_specific_idx=k_specific_idx+1

                    batch, channel, h, w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
                    x_i=torch.transpose(x_i,1,3).contiguous().view(batch*w*h,channel)
                    #x_i=logmap0(x_i,k=c_i)
                    f.append ( torch.transpose(x_i.contiguous().view(batch,w,h,channel),1,3) ) # (batch,channel,h,w)

                f=torch.cat(f,dim=1) # (n,c,h,w)
                f = F.adaptive_avg_pool2d(f, (1,1))
                f = f.view(f.shape[0], -1)


                data=torch.zeros(param[4],param[3]).cuda()
                for ii in range(batch):
                    l=label[ii]
                    data[l,:]=data[l,:]+f[ii,:].detach()
                data=data/self.args.k_spt
                c = self.controller(torch.sum(data,dim=0),torch.sum(data,dim=0))


                #w1, b1 = classifier[0], classifier[1]
                w1 = classifier[0]
                b1 = classifier[1]
                #logit=F.linear(f, w1, bias=b1)
                f_expand=f.repeat(1,class_num)
                f_expand=f_expand.reshape(f.shape[0],class_num,param[3])
                w1_expand=w1.repeat(f.shape[0],1,1)

                logit=w1_expand*f_expand
                logit=logit.reshape(f.shape[0],class_num,compoent_num,component_channel)
                logit=torch.sum(logit,3)
                logit=logit.reshape(f.shape[0]*class_num,compoent_num)

                att_w1, att_b1= classifier[2], classifier[3]
                logit_att=F.linear(logit, att_w1, att_b1)
                logit_att=F.softmax(logit_att)
                logit=logit*logit_att*class_num  #(f.shape[0]*class_num,compoent_num)
                logit=torch.sum(logit,1).reshape(f.shape[0],class_num)
                logit=logit+b1

                loss = F.cross_entropy(logit, label)

                specific_parameters_grad = torch.autograd.grad(loss, specific_parameter, retain_graph=True)
                classifier_grad = torch.autograd.grad(loss, classifier, retain_graph=True,allow_unused= True)
                specific_k_grad = torch.autograd.grad(loss, specific_k, allow_unused= True)

                num_specific_parameters=len(specific_parameters_grad)
                num_classifier_parameters=len(classifier_grad)
                num_k=len(specific_k_grad)  
  


                per_step_task_embedding = []

                for j in range(len(specific_parameters_grad)):
                    per_step_task_embedding.append(specific_parameter[j].mean())   
                    per_step_task_embedding.append(specific_parameters_grad[j].mean())
                for j in range(len(classifier_grad)):
                    per_step_task_embedding.append(classifier[j].mean()) 
                    per_step_task_embedding.append(classifier_grad[j].mean())
                for j in range(len(specific_k_grad)):
                    per_step_task_embedding.append(specific_k[j].mean())  
                    per_step_task_embedding.append(specific_k_grad[j].mean())



                #per_step_task_embedding.append((torch.ones(1)*(1)).cuda().mean())

                per_step_task_embedding = torch.stack(per_step_task_embedding)
                per_step_task_embedding = per_step_task_embedding.unsqueeze(0)

                generated_weight= self.generate_l2f(per_step_task_embedding)
                generated_weight = torch.squeeze(generated_weight)


                new_specific_parameter=[]
                new_classifier=[]
                new_specific_k=[]

                for ii in range(num_specific_parameters+num_classifier_parameters+num_k):
                    if ii < num_specific_parameters:
                        new_specific_parameter.append(specific_parameter[ii]*generated_weight[ii])
                    elif ii >=num_specific_parameters and ii < (num_specific_parameters+num_classifier_parameters):
                        new_classifier.append(classifier[ii-num_specific_parameters]*generated_weight[ii])
                    elif ii >=(num_specific_parameters+num_classifier_parameters) and ii < (num_specific_parameters+num_classifier_parameters+num_k):
                        new_specific_k.append(specific_k[ii-num_specific_parameters-num_classifier_parameters]*generated_weight[ii]+ c[ii-num_specific_parameters-num_classifier_parameters])

        return  new_specific_parameter, new_classifier, new_specific_k






    def zero_grad(self, vars):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            for p in vars:
                if p.grad is not None:
                    p.grad.zero_()

    def specific_parameters_finetune(self):
        return self.specific_parameter.extend(self.classifier)

    def classifier_parameters(self):
        return self.classifier

    def classifier_generators(self):
        return self.classifier_generator

    def specific_parameters_noclassifier(self):
        l=len(self.specific_parameter)
        return self.specific_parameter[0:l-1]

    def specific_parameters(self):
        return self.specific_parameter

    def shared_parameters(self):
        return self.shared_parameter

    def specific_ks(self):
        return self.specific_k

    def shared_ks(self):
        return self.shared_k

    def specific_bns(self):
        return self.specific_bn        

    def shared_bns(self):
        return self.shared_bn   

    def shared_lrs(self):
        return self.shared_lr
  
    def shared_klrs(self):
        return self.shared_klr

    def shared_classifier_lrs(self):
        return self.classifier_lr


    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param


