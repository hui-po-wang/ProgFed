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

import time

from pthflops import count_ops

import pdb
import numpy as np

import models
from utils import AverageMeter, Statistics, accuracy, Parser, LearningScheduler, UpdateScheduler

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)

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
    arch_choices = ['resnet18', 'resnet152', 'vgg16_bn', 'vgg19_bn']

    if args.dataset == 'cifar100':
        x = torch.rand(args.batch_size, 3, 32, 32).to(device)
        y = torch.ones(args.batch_size).to(device).long()
    train_set, train_loader, val_set, val_loader = prepare_data(args)

    for ac in arch_choices:
        Network = getattr(models, ac)
        model_server = Network(args).to(device)
        crit = nn.CrossEntropyLoss().to(device)

    
        for i in range(4):
            model_server.set_submodel(i)
            submodel = model_server.gen_submodel().to(device)
            num_p = sum([p.numel() for p in submodel.parameters()])
            print(num_p)
            opt = torch.optim.SGD(submodel.parameters(), args['lr_scheduler']['lr'],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
            
            # dummy operations to wake up GPUs
            pred = submodel(x)
            loss = crit(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
           
            accu_time = 0
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                start_time = time.time()
                pred = submodel(imgs)
                loss = crit(pred, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                accu_time += time.time() - start_time
            print(f'Training time {accu_time}')

            ops, _ = count_ops(submodel, x, verbose=False)
            print(f'Dataset {args.dataset}, Arch {ac}, stage {i}: {ops}')


          
if __name__ == '__main__':
    args = parse_args()

    main(args)
