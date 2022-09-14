import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
from datasets.samplers import CategoriesSampler
from utils import compute_confidence_interval

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total,', total_num, 'Trainable,', trainable_num)
    return {'Total': total_num, 'Trainable': trainable_num}


def main(config):
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']+'curvautre-1_lr0.05_component16_resnet12projection'
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # val
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'],
                                    **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=8, pin_memory=True)
        utils.log('val dataset: {} (x{}), {}'.format(
                val_dataset[0][0].shape, len(val_dataset),
                val_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False


    #### Model and Optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])


    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model,
            config['optimizer'], **config['optimizer_args'])

    get_parameter_number(model)
    ########
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(1, max_epoch + 1 + 1):
        if epoch == max_epoch + 1:
            if not config.get('epoch_ex'):
                break
            train_dataset.transform = train_dataset.default_transform
            train_loader = DataLoader(
                    train_dataset, config['batch_size'], shuffle=True,
                    num_workers=8, pin_memory=True)

        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, label in tqdm(train_loader, desc='train', leave=False):
            data, label = data.cuda(), label.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)
            #print('train loss',loss)
            optimizer.zero_grad()
            loss.backward()

            
            for param in model.trainable_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
            

            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None

        # eval
        if eval_val:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                
                aves['vl'].add(loss.item())
                aves['va'].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)


        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_classifier_mini.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

