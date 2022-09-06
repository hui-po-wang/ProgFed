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
import idx2numpy
from random import shuffle
from tqdm import tqdm

# import local library
import models
from fl_utils import (adjust_learning_rate, set_model, update_model, compute_client_gradients, 
    VirtualWorker, loss_prox, _zero_weights, adjust_gradient_by_scaffold, update_client_state, update_server_state, update_model_global_optim)
from utils import AverageMeter, Statistics, accuracy, Parser, LearningScheduler, UpdateScheduler, Cifar100_FL_Dataset, EMNIST_FL_Dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)
    parser.add_argument('-seed', '--seed', default=None)

    parser.add_argument('-data-path', '--data-path', default='/huipo/datasets', type=str)
    parser.add_argument('-download', '--download', action='store_true')

    parser.add_argument('-save_path', '--save_path', default='./saves', type=str)

    # if start-epoch != 1, load the pretrained model
    parser.add_argument('-start-epoch', '--start-epoch', default=1, type=int)
    parser.add_argument('-start-model', '--start-model', default='./', type=str)

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

def prepare_workers(args, trainset, use_cuda, split_in=False):
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {'pin_memory': True} if use_cuda else {}

    # Create number of virtual workers that will act as clients
    workers = {}
    for i in range(args.n_client):
        workers[i] = VirtualWorker(i)

    # If split has been created outside the function, just assign it to the workers
    if split_in:
        if args.n_client != len(trainset):
            raise ValueError(f'#client ({args.n_client}) != #training splits {len(trainset)}.')
        for i in range(args.n_client):
            workers[i].set_loader(torch.utils.data.DataLoader(trainset[i],
                                batch_size=args.batch_size, shuffle=True, **kwargs))
    else: # divide the training set according to the noniid option
        if args.noniid:
            data_id, _ = noniid(trainset, args.n_client, args.shard_per_user)
            print(f'non-iid split shape: {len(data_id)}x{data_id[0].shape}')
            for i in range(args.n_client):
                workers[i].set_loader(torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, data_id[i]),
                                        batch_size=args.batch_size, shuffle=True, **kwargs))
        else:
            data_id = list(range(len(trainset)))
            shuffle(data_id)
            n_sample_per_client = int(len(trainset) / args.n_client)
            for i in range(args.n_client):
               workers[i].set_loader(torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, data_id[i*n_sample_per_client:i*n_sample_per_client+n_sample_per_client]),
                                batch_size=args.batch_size, shuffle=True, **kwargs))

    return workers
    
def train(args, global_optim, full_model, subnet_server, subnet, state_server, metric,
    device, workers, epoch, buffer, state_buffer, lr_scheduler, warmup=False):
    subnet.train()

    #current_lr = max(args['lr_scheduler']['lr'] * (1 + np.cos(np.pi * (epoch-1) / (args.epochs-1) ) ) / 2 , 1e-6)
    #current_lr = args.lr

    client_samples = list(range(args.n_client))

    buffer['gradient_data'] = []
    buffer['gradient_rec1'] = []
    buffer['gradient_rec2'] = []
    buffer['gradient_rec3'] = []

    state_buffer['state_data'] = []

    shuffle(client_samples)
    for id_client in client_samples[:args.n_update_client]:
        current_worker = workers[id_client]
        current_data_loader = current_worker.loader

        # mimic sending model weights to clients
        start_time = time.time()
        set_model(subnet_server, subnet.module, args)
        print("--- %s seconds for copy submodel---" % (time.time() - start_time))

        optimizer = current_worker.opt
        #adjust_learning_rate(optimizer, current_lr)
        if not warmup:
            lr_scheduler.set_opt(optimizer)
        
        for epoch_client in range(args.epoch_client):
            epoch_time = time.time()
            for batch_idx, (data, target) in enumerate(current_data_loader): # <-- now it is a distributed dataset
                #start_time = time.time()
                data, target = data.to(device), target.to(device)
                #print("--- %s seconds for preparing data---" % (time.time() - epoch_time))
                #start_time = time.time()
                
                output = subnet(data)
                if args.optimization == 'fedprox':
                    loss = metric(output, target) + args.mu_loss_prox * loss_prox(subnet_server , subnet.module, device)
                else:
                    loss = metric(output, target)

                if loss < 10:
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                #print("--- %s seconds for one training---" % (time.time() - start_time))

                if global_optim['optim_init']:
                    global_optim['optim'].zero_grad()
                    loss_global = metric(subnet_server(data), target) * 0
                    loss_global.backward()
                    global_optim['optim'].step()
                    global_optim['optim_init'] = False

                if batch_idx % args.log_interval == 0:
                    for param_group in optimizer.param_groups:
                        current_learning_rate = param_group['lr']

                    print('Train Epoch: {}, Client: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.4f}'.format(
                        epoch, id_client, batch_idx * args.batch_size, len(current_data_loader) * args.batch_size,
                        100. * batch_idx / len(current_data_loader) / 100, loss.item(), current_learning_rate ))
            print("--- %s seconds for one local epoch---" % (time.time() - epoch_time))
        
        #start_time = time.time()
        compute_client_gradients(subnet_server, subnet.module, buffer, args)

    update_model_global_optim(global_optim['optim'], subnet_server, buffer, device, args)
    if not warmup:
        lr_scheduler.step()

def test(args, model, device, test_loader, result):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    result.append( 100. * correct / len(test_loader.dataset) )

    model.train()

def create_server_opt(subnet_server, args):
    global_optim = {}
    if args.optimization == 'fedadam':
        #global_optim['optim'] = optim.Adam(params=subnet_server.parameters(), lr=0.1, weight_decay=args.weight_decay)
        global_optim['optim'] = optim.Adam(params=subnet_server.parameters(), lr=args.global_lr)
    else:
        global_optim['optim'] = optim.SGD(params=subnet_server.parameters(), lr=args.global_lr)
    global_optim['optim_init'] = True
    return global_optim

def main(args):
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)


    # data
    trainset, test_loader, split_in = prepare_data(args, use_cuda)
    # workers
    workers = prepare_workers(args, trainset, use_cuda, split_in)
    # Initialize the model
    Network = getattr(models, args.arch)
    model_server = Network(args).to(device)

    n_param_model = 0
    for parameter in model_server.parameters(): n_param_model += parameter.nelement()
    print("# of model parameters: %d"%n_param_model)

    if args.start_epoch != 1:
        model_load_tmp = torch.load(args.start_model)
        model.load_state_dict(model_load_tmp["state_dict"] , strict=False)
        model_server.load_state_dict(model_load_tmp["state_dict"] , strict=False)
        result = list(model_load_tmp["result"].numpy()[:-1])

    if args.strategy == 'baseline':
        layer_cnt = 3
    else:
        layer_cnt = 0

    """Dynamic updates, not used so far"""
    if args.update_strategy == None:
        update_scheduler = UpdateScheduler(args.update_cycle, num_stages=args.num_stages, update_strategy=None)
    else:
        update_scheduler = UpdateScheduler(model_server.return_stage_parameters(), num_stages=args.num_stages, update_strategy=args.update_strategy)

    print(update_scheduler)

    metric = nn.CrossEntropyLoss()
    model_server.set_submodel(layer_cnt)
    print(model_server)
    # define subnets, which will be transmitted during training
    subnet_server = model_server.gen_submodel().to(device)

    global_optim = create_server_opt(subnet_server, args)
    
    state_server = None
    subnet = torch.nn.DataParallel(copy.deepcopy(subnet_server).to(device))

    # initialize worker on every client
    for i in range(args.n_client):
        workers[i].set_opt(optim.SGD(params=subnet.parameters(), lr=args['lr_scheduler']['lr'], momentum=args.momentum, weight_decay=args.weight_decay))

    lr_scheduler = LearningScheduler(args)

    # log
    writer = SummaryWriter(os.path.join('runs/', args.arch, args.name))

    result = []
    accu_cost = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs + 1)):
        # Scheduling for progressive training
        if (args.strategy != 'baseline' and epoch != 0 and 
                epoch == update_scheduler[layer_cnt] and layer_cnt < args.num_stages-1):
            layer_cnt += 1
            model_server.set_submodel(layer_cnt)
            if args.strategy != 'svcca':
                subnet_server = model_server.gen_submodel().to(device)
                subnet = torch.nn.DataParallel(copy.deepcopy(subnet_server).to(device))
            else:
                subnet_server.ind = layer_cnt
                subnet = torch.nn.DataParallel(copy.deepcopy(subnet_server).to(device))
            print(f'{args.strategy}, {layer_cnt}')
            print(subnet_server)

            # Handling warm-up
            if args.warmup and args.strategy != 'layerwise':
                # initialize the global optimizer for warm-up
                global_optim = create_server_opt(subnet_server, args)
                for j in range(args.n_client):
                    workers[j].set_opt(optim.SGD(params=subnet.module.lastest_parameters(), lr=args['lr_scheduler']['lr'],
                    #workers[j].set_opt(optim.SGD(params=subnet.lastest_parameters(), lr=10*args['lr_scheduler']['lr'],
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay))
                for w_i in range(args.warmup_epochs):
                    print(f'{w_i}th warmup')
                    cur_cost += sum(p.numel() for p in subnet_server.lastest_parameters())
                    if args.quantize_option != 'none': 
                        accu_cost += args.n_update_client * (cur_cost*args.quantize_bits/8/1000/1000)
                    else:
                        accu_cost += args.n_update_client * (cur_cost*4/1000/1000)
                
                    train(args, global_optim, model_server, subnet_server, subnet, state_server, metric, device, workers, epoch, buffer, state_buffer, lr_scheduler, warmup=True)
            
            # Prepare training new sub-models and re-init the global optimizer
            global_optim = create_server_opt(subnet_server, args)
            if args.strategy == 'layerwise':
                for i in range(args.n_client):
                    workers[i].set_opt(optim.SGD(params=subnet.module.lastest_parameters(), lr=args['lr_scheduler']['lr'],
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay))
            elif args.strategy == 'mixed' or args.strategy == 'dense':
                raise NotImplementedError()
            elif args.strategy in ['progressive', 'partial', 'svcca']:
                for i in range(args.n_client):
                    workers[i].set_opt(optim.SGD(params=subnet.module.trainable_parameters(), lr=args['lr_scheduler']['lr'],
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay))
            else:
                raise NotImplementedError()
            #lr_scheduler.set_opt(opt)

        # record communication cost
        if args.strategy == 'layerwise':
            cur_cost = sum(p.numel() for p in subnet_server.lastest_parameters())
        elif args.strategy == 'mixed':
            cur_cost = (sum(p.numel() for p in subnet_server.trainable_parameters()) + sum(p.numel() for p in model_server.fc.parameters()))
        else:
            cur_cost = subnet_server.return_num_parameters()
        
        # megabytes
        if args.quantize_option != 'none': 
            accu_cost += args.n_update_client * (cur_cost*args.quantize_bits/8/1000/1000)
        else:
            accu_cost += args.n_update_client * (cur_cost*4/1000/1000)

        buffer = {}
        state_buffer = {}
        train(args, global_optim, model_server, subnet_server, subnet, state_server, metric, device, workers, epoch, buffer, state_buffer, lr_scheduler)
        if epoch % args.test_interval == 0:
            #test(args, model_server, device, test_loader, result)
            start_time = time.time()
            test(args, subnet_server, device, test_loader, result)
            print("--- %s seconds for test---" % (time.time() - start_time))
            writer.add_scalar('Metric/acc-epoch', result[-1], epoch)
            writer.add_scalar('Metric/acc-cost', result[-1], accu_cost)
            writer.add_scalar('Debug/layer_cnt', layer_cnt, epoch)
            writer.add_scalar('Debug/lr', lr_scheduler.get_lr(), epoch)

        if args.save_model and epoch % args.save_interval == 1 and epoch != 1:
            file_name = os.path.join(args.train_dir, 'model_%04d.tar'%epoch )
            res = torch.from_numpy(np.array(result))

            torch.save({
                'args': vars(args),
                'epoch': epoch,
                'state_dict': model_server.state_dict(),
                #'optim_dict': opt.state_dict(),
            }, file_name)

    if (args.save_model):
        file_name = os.path.join(args.train_dir, 'model_last.tar')
        res = torch.from_numpy(np.array(result))

        torch.save({
                'args': vars(args),
                'epoch': epoch,
                'state_dict': model_server.state_dict(),
                #'optim_dict': opt.state_dict(),
            }, file_name)
    writer.close()

if __name__ == '__main__':
    args = parse_args()

    main(args)
