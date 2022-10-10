import argparse
import os
import yaml
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.models as torch_models

import numpy as np

import models
from utils import AverageMeter, Statistics, accuracy, Parser, LearningScheduler, UpdateScheduler

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)
    parser.add_argument('-seed', '--seed', default=None)

    parser.add_argument('-data-path', '--data-path', default='/p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/datasets', type=str)
    parser.add_argument('-download', '--download', action='store_true')

    parser.add_argument('-save_path', '--save_path', default='./saves', type=str)


    args = parser.parse_args()
    with open(args.cfg, 'r') as stream:
        settings = yaml.safe_load(stream)
    args = Parser(args, settings)
    args.name = os.path.basename(args.cfg).split('.')[0]

    # used for keeping all model weights and the configuration file, etc.
    args.train_dir = os.path.join(args.save_path, args.arch, args.name)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    print(args)
    return args

def prepare_data(args):
    if args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_set = torchvision.datasets.ImageNet(args.data_path, split='train', 
            download=args.download,
            transform=train_trans)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1, pin_memory=True)

        val_set = torchvision.datasets.ImageNet(args.data_path, split='val', 
            download=args.download,
            transform=val_trans)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1, pin_memory=True)

    elif args.dataset == 'cifar10':
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = torchvision.datasets.CIFAR10(
            root=args.data_path, train=True, download=args.download, transform=train_trans)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

        val_set = torchvision.datasets.CIFAR10(
            root=args.data_path, train=False, download=args.download, transform=val_trans)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    elif args.dataset == 'cifar100':
        # copy from https://github.com/weiaicunzai/pytorch-cifar100/blob/2149cb57f517c6e5fa7262f958652227225d125b/utils.py#L166
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_set = torchvision.datasets.CIFAR100(
            root=args.data_path, train=True, download=args.download, transform=train_trans)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

        val_set = torchvision.datasets.CIFAR100(
            root=args.data_path, train=False, download=args.download, transform=val_trans)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'mnist':
        pass
    else:
        raise NotImplementedError()

    return train_set, train_loader, val_set, val_loader

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize the model
    Network = getattr(models, args.arch)
    model_server = Network(args).to(device)
    
    # This code is assumed to fine-tune a well-trained model
    layer_cnt = 3

    if args.update_strategy == None:
        update_scheduler = UpdateScheduler(args.update_cycle, num_stages=4, update_strategy=None)
    else:
        update_scheduler = UpdateScheduler(model_server.return_stage_parameters(), num_stages=4, update_strategy=args.update_strategy)

    print(update_scheduler)
    model_server.set_submodel(layer_cnt)

    ##################
    #
    # Load model weights
    #
    ##################
    model_load_tmp = torch.load(os.path.join(args.train_dir, 'model_epoch_199.tar'))
    model.load_state_dict(model_load_tmp["state_dict"])

    print(model_server)
    submodel = model_server.gen_submodel().to(device)
    crit = nn.CrossEntropyLoss().to(device)

    #opt = torch.optim.SGD(submodel.parameters(), args['lr_scheduler']['lr'],
    opt = torch.optim.SGD(submodel.parameters(), lr=1e-4,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #lr_scheduler = LearningScheduler(**args.lr_scheduler)
    #lr_scheduler.set_opt(opt)

    # Initialize the dataset and the loader
    train_set, train_loader, val_set, val_loader = prepare_data(args)

    # loggers
    writer = SummaryWriter(os.path.join('runs/', args.arch, args.name))
    stats = Statistics()
    accu_cost = 0
    warmup_trigger = False

    for i_iter in tqdm(range(args.epochs)):
        # meters for inner-loops
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        losses = AverageMeter('Losses', ':6.2f')

        # record communication cost
        if args.strategy == 'layerwise':
            accu_cost += sum(p.numel() for p in submodel.lastest_parameters())
        elif args.strategy == 'mixed':
            accu_cost += (sum(p.numel() for p in submodel.trainable_parameters()) + sum(p.numel() for p in model_server.fc.parameters()))
        else:
            accu_cost += submodel.return_num_parameters()

        submodel.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            if args.strategy == 'mixed':
                loss = 0.5 * crit(submodel(x), y) + 0.5 * crit(model_server(x), y)
            elif args.strategy == 'dense':
                preds = submodel(x)
                preds = preds[:-1] if len(preds) == args.num_stages else preds
                loss = 0
                for p in preds:
                    loss += crit(p, y)
                loss += crit(model_server(x), y)
            else:
                pred = submodel(x)
                loss = crit(pred, y)
            losses.update(loss.item(), x.size(0))

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'Iter {i}: loss={losses.avg}.')
        
        writer.add_scalar('Metric/loss-epoch', losses.avg, i_iter)
        writer.add_scalar('Metric/loss-cost', losses.avg, accu_cost)
        writer.add_scalar('Debug/lr', lr_scheduler.get_lr(), i_iter)
        writer.add_scalar('Debug/layer_cnt', layer_cnt, i_iter)
        stats.add('losses', losses.avg)
        stats.add('accu_cost', accu_cost)
        stats.add('lr', lr_scheduler.get_lr())

        # update learning rate
        #lr_scheduler.step()

        # TO-DO record accuracy, evaluate on val, and save models
        if i_iter == 0 or (i_iter+1) % args.save_freq == 0:
            submodel.eval()
            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(val_loader)):
                    x = x.to(device)
                    y = y.to(device)

                    if args.strategy == 'dense':
                        pred = submodel(x)[-1]
                    elif args.strategy == 'random':
                        pred = model_server(x)
                    else:
                        pred = submodel(x)
                        
                    if args.dataset == 'imagenet':
                        acc1, acc5 = accuracy(pred, y, topk=(1, 5))
                        acc1, acc5 = acc1[0], acc5[0]
                    else:
                        acc1 = accuracy(pred, y)
                        acc1 = acc1[0]

                    top1.update(acc1.item(), x.size(0))
                    if args.dataset == 'imagenet':
                        top5.update(acc5.item(), x.size(0))
    
            stats.add('acc@1', (top1, i_iter, accu_cost))
            writer.add_scalar('Acc/acc@1-epoch', top1.avg, i_iter)
            writer.add_scalar('Acc/acc@1-cost', top1.avg, accu_cost)
            if args.dataset == 'imagenet':
                stats.add('acc@5', (top5, i_iter, accu_cost))
                writer.add_scalar('Acc/acc@5-epoch', top5.avg, i_iter)
                writer.add_scalar('Acc/acc@5-cost', top5.avg, accu_cost)
                print(f'Validation Epoch {i_iter}: acc@1={top1.avg} acc@5={top5.avg}')
            else:
                print(f'Validation Epoch {i_iter}: acc@1={top1.avg}')

            file_name = os.path.join(args.train_dir, 'model_epoch_ft_{}.tar'.format(i_iter))
            torch.save({
                'args': vars(args),
                'epoch': i_iter,
                'state_dict': model_server.state_dict(),
                'optim_dict': opt.state_dict(),
                'stats': stats
            }, file_name)   
    writer.close()  
          
if __name__ == '__main__':
    args = parse_args()

    main(args)