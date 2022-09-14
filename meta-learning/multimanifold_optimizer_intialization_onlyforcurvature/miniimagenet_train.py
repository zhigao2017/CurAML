import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        #(input_channel, output_channel, component_num, specific_num, Riemannian_component_num, max_pool_padding, No.layer )
        ('block', [3, 64, 16, 16, 16, 0, '1']),
        ('block', [64, 160, 16, 16, 16, 0, '2']),
        ('block', [160, 320, 16, 16, 16, 0, '3']),
        ('block', [320, 640, 16, 16, 16, 0, '4']),
        #('distance',w,h,compoent_num,all_channel,class_num, Riemannian_component_num)
        ('distance', [5, 5, 16, 640, 5, 16])
    ]

    print('config',config)

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=500, resize=args.imgsz)
    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    count=0
    for epoch in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry,step,epoch)
            count=count+1

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 499 == 0:  # evaluation
                torch.save(maml.net.regularizer.state_dict(), 'save_n+1dimension_3elayer_sigmoid1e-3-1e5_onlyforcurvature/updatelr_'+str(1)+'_backbonelr_'+str(1)+'_backbonelr_'+str(1)+'_epoch_'+str(epoch)+'_step_'+str(step)+'.pth')



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60)
    argparser.add_argument('--multi_step_loss_num_epochs', type=int, help='epoch number', default=15)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    argparser.add_argument('--min_meta_lr', type=float, help='meta-level outer learning rate', default=1e-5)      
    argparser.add_argument('--backbone_lr', type=float, help='meta-level outer learning rate', default=1e-5)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    argparser.add_argument('--k_lr', type=float, help='task-level inner update learning rate', default=1e-5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=4)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--divide', type=float, help='divide for the curvature', default=1e-1)

    args = argparser.parse_args()  

    print('args',args)

    main()
