# import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# import python library
import os
import random
import numpy as np
import argparse
import zlib
import copy
import sys
import yaml
import time
from random import shuffle
from tqdm import tqdm

# import local library
import models
from fl_utils import (adjust_learning_rate, set_model, update_model, compute_client_gradients, 
    VirtualWorker, loss_prox, _zero_weights, adjust_gradient_by_scaffold, update_client_state, update_server_state)
from utils import AverageMeter, Statistics, accuracy, Parser, LearningScheduler, UpdateScheduler, Cifar100_FL_Dataset, EMNIST_FL_Dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)
    parser.add_argument('-seed', '--seed', default=None)

    parser.add_argument('-data-path', '--data-path', default='/p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/datasets', type=str)
    parser.add_argument('-download', '--download', action='store_true')

    parser.add_argument('-save_path', '--save_path', default='./saves', type=str)

    # if start-epoch != 1, load the pretrained model
    parser.add_argument('-start-epoch', '--start-epoch', default=1, type=int)
    parser.add_argument('-start-model', '--start-model', default='./saves/model_last.tar', type=str)

    args = parser.parse_args()
    with open(args.cfg, 'r') as stream:
        settings = yaml.safe_load(stream)
    args = Parser(args, settings)
    args.name = os.path.basename(args.cfg).split('.')[0]

    # used for keeping all model weights and the configuration file, etc.
    args.train_dir = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    print(args)
    return args

def prepare_data(args, use_cuda):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    split_in = False
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(
                size=32,
                padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) ),
        ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(args.data_path, train=True, transform=transform_train, download=args.download)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_path, train=False, transform=transform_test, download=args.download),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'cifar100':
        split_in = True
        transform_train = transforms.Compose([
            transforms.RandomCrop(
                size=24,
                padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) ),
        ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = []
        for i in range(args.n_client):
            dset_tmp = Cifar100_FL_Dataset(args.data_path, i, transform=transform_train)
            trainset.append(dset_tmp)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_path, train=False, transform=transform_test, download=args.download),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        trainset = datasets.MNIST(args.data_path, train=True, transform=transform_train, download=args.download)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_path, train=False, transform=transform_test, download=args.download),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'emnist':
        split_in = True
        data_num = np.load(f"{args.data_path}/EMNIST/num.npy").astype(np.uint)
        data_start = np.array([0] + list(np.load(f"{args.data_path}/EMNIST/num.npy"))).astype(np.uint)
        for i in range(1,len(data_start)):
            data_start[i] = data_start[i] + data_start[i-1]
    
        train_data_ubyte = idx2numpy.convert_from_file(f"{args.data_path}/EMNIST/emnist-byclass-train-images-idx3-ubyte")
        train_label_ubyte = idx2numpy.convert_from_file(f"{args.data_path}/EMNIST/emnist-byclass-train-labels-idx1-ubyte")
        test_data_ubyte = idx2numpy.convert_from_file(f"{args.data_path}/EMNIST/emnist-byclass-test-images-idx3-ubyte")
        test_label_ubyte = idx2numpy.convert_from_file(f"{args.data_path}/EMNIST/emnist-byclass-test-labels-idx1-ubyte")

        transform_train = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

        transform_test = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    
        test_loader = torch.utils.data.DataLoader(
            EMNIST_FL_Dataset(test_data_ubyte[:77483], test_label_ubyte[:77483], transform=transform_test ),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        trainset = []
        for i in range(args.n_client):
            dset_tmp = EMNIST_FL_Dataset(train_data_ubyte[data_start[i]:data_start[i]+data_num[i]], train_label_ubyte[data_start[i]:data_start[i]+data_num[i]], transform=transform_train )
            trainset.append(dset_tmp)
    else:
        raise NotImplementedError()

    return trainset, test_loader, split_in

def test(args, model, device, test_loader, result):
    model.eval()
    correct = [0 for _ in range(args.num_stages)]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.dense_forward(data)
            for i in range(args.num_stages):
                pred = output[i].argmax(1, keepdim=True) # get the index of the max log-probability 
                correct[i] += pred.eq(target.view_as(pred)).sum().item()

    for i in range(args.num_stages):
        print('Stage {i} Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main(args):
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)


    # data
    trainset, test_loader, split_in = prepare_data(args, use_cuda)

    Network = getattr(models, args.arch)
    model_server = Network(args).to(device)

    n_param_model = 0
    for parameter in model_server.parameters(): n_param_model += parameter.nelement()
    print("# of model parameters: %d"%n_param_model)

    if args.start_epoch != 1:
        model_load_tmp = torch.load(args.start_model)
        model.load_state_dict(model_load_tmp["state_dict"])
        model_server.load_state_dict(model_load_tmp["state_dict"])
        result = list(model_load_tmp["result"].numpy()[:-1])

    test(args, model_server, device, test_loader, result)
  

if __name__ == '__main__':
    args = parse_args()

    main(args)