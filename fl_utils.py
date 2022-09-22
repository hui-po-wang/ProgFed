import numpy as np
import random

# import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from quantization import cosine_quantization, cosine_dequantization
from quantization import linear_quantization, linear_dequantization 
from quantization import add_dp_noise

import sys

class VirtualWorker():
    def __init__(self, wid):
        self.wid = wid
        #self.dset = None 
        self.state = None
        self.loader = None
        self.opt = None

    def set_loader(self, loader):
        self.loader = loader

    def set_opt(self, opt):
        self.opt = opt

    def init_state(self, state):
        self.state = state
        self.state.apply(_zero_weights)
        self.state.requires_grad = False
        self.state.to('cpu')

def _zero_weights(m):
    for p in m.parameters():
        torch.nn.init.constant_(p, 0)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_model(src, dst, args):
    if args.quantize_option == 'dp-server':
        raise NotImplementedError()
    else:
        for s, d in zip( src.parameters() , dst.parameters() ):
            d.data = s.data.detach().clone()

def update_model_global_optim(global_optim, model, buffer, device, args):
    q_option = args.quantize_option
    n_bits = args.quantize_bits
    sparse = args.sparse

    global_optim.zero_grad()

    ###
    # buffer[type_data][num_cients][name_layers]
    ###
    for i in range(len(buffer['gradient_data'])):
        for k, p in model.named_parameters():
            weight = 600 * len(buffer['gradient_data'])
            grad_out = 0
            n_nan = 0

            if not k in buffer['gradient_data'][i].keys():
                continue

            data = buffer['gradient_data'][i][k]

            if q_option == 'dp-client':
                raise NotImplementedError()
            elif q_option == 'none':
                data = data.cuda()
            elif q_option == 'cosine':
                data = cosine_norm_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k])
            elif q_option == 'linear':
                data = linear_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k], buffer['gradient_rec3'][i][k].to(device), args.quantize_hadamard)
            elif q_option == 'comb':
                data = combine_dequantization(data, buffer['gradient_rec3'][i][k], n_bits, args.quantize_bits_low, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k])
            elif q_option == 'kmeans':
                data = kmeans_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], device)
            else:
                print("Unexpected quantization method:", sys.exc_info()[0])
                raise RuntimeError from OSError

            if sparse == 1:
                grad_out += - data * 600 / weight
            elif sparse > 0 and sparse < 1:
                mask = (torch.rand(data.size()) < sparse).type(data.type()).cuda()
                grad_out += - data * mask  / sparse * 600 / weight
            else:
                print("Unexpected sparsification ratio:", sys.exc_info()[0])
                raise RuntimeError from OSError

            if args.n_update_client > n_nan:
                p.grad.add_( -grad_out.cuda() )

    global_optim.step()
    
def update_model(model, buffer, args):
    q_option = args.quantize_option
    n_bits = args.quantize_bits
    sparse = args.sparse

    for k, p in model.named_parameters(): 
        weight = 600 * len(buffer['gradient_data'])
        grad_out = 0
        n_nan = 0
        
        #print(k)
        for i in range(len(buffer['gradient_data'])):
            # TODO: check whether the name of the current grad exists in the buffer
            if not k in buffer['gradient_data'][i].keys():
                break
            data = buffer['gradient_data'][i][k]
            
            if q_option == 'dp-client':
                pass
            elif q_option == 'none':
                data = data
            elif q_option == 'cosine':
                data = cosine_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k])
            elif q_option == 'linear':
                data = linear_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k], buffer['gradient_rec3'][i][k], args.quantize_hadamard)
            else:
                print("Unexpected quantization method:", sys.exc_info()[0])
                raise RuntimeError from OSError

            if sparse == 1:
                grad_out += - data * 600 / weight
            elif sparse > 0 and sparse < 1 and not q_option == 'dp-client':
                mask = (torch.rand(data.size()) < sparse).type(data.type()).cuda()
                grad_out += - data * mask  / sparse * 600 / weight
            else:
                raise ValueError(f'Not valid sparsification ratios ({sparse}) or options {q_option}')

        if args.n_update_client > n_nan:
            p.data.add_( grad_out.cuda() )

def update_client_state(state_server, state_client, buffer, state_buffer, lr_local, K, device):
    '''We update client states according to Option 2 in Eq.4 of the SCAFFOLD paper
       That is, c_i - c + 1/(K*eta_l)*(x-y_i)
    '''
    gradient_data = buffer['gradient_data'][-1]
    state_data = {}
    for m1, m2 in zip(state_server.named_parameters(), state_client.named_parameters()):
        assert m1[0] == m2[0]
        assert m1[1].shape == m2[1].shape

        # graident_data stores the value of init_weights - new_weights
        state_diff = - m1[1] + gradient_data[m1[0]].to(device) * (1/(lr_local * K))
        state_data[m1[0]] = state_diff.detach()
        m2[1].data.add_(state_diff)

    state_buffer['state_data'].append(state_data)


def update_server_state(state_server, state_buffer, device, args):
    state_data = state_buffer['state_data']
    for n, p in state_server.named_parameters():
        state_out = 0
        for i in range(len(state_data)):
            assert n in state_data[i].keys()

            state_out += state_data[i][n]

        assert len(state_data) > 0
        state_out /= len(state_data)
        p.data.add_(state_out.to(device))


def adjust_gradient_by_scaffold(subnet, state_server, state_client, lr, device):
    for m1, m2, m3 in zip(subnet.named_parameters(),
        state_server.named_parameters(), state_client.named_parameters()):

        assert m1[0] == m2[0] and m2[0] == m3[0]
        assert m1[1].shape == m2[1].shape and m2[1].shape == m3[1].shape
        # server state - client state in accordance with Eq.3 in SCAFFOLD
        var = m2[1] - m3[1]
        
        m1[1].grad += (var.to(device)*lr)

def compute_client_gradients(model, model_new, buffer, args):
    q_option = args.quantize_option
    q_clip = args.quantize_clip
    n_bits = args.quantize_bits

    gradient_data = {}
    gradient_rec1 = {}
    gradient_rec2 = {}
    gradient_rec3 = {}

    if q_option in ['dp_both', 'dp_up']:
        const_c = np.sqrt(2*np.log(1.25/args.delta))
        if args.dataset == 'cifar100':
            min_m = 100
        else:
            raise NotImplementedError(f'Unknown datasets {args.dataset} for DP exps.')

        delta_su = 2*q_clip/min_m
        sigma_u = const_c * delta_su / args.epsilon
        
    for m1, m2 in zip( model.named_parameters() , model_new.named_parameters() ):
        assert m1[0] == m2[0]
        assert m1[1].shape == m2[1].shape
        #print(m1[0], m2[0])
        tmp = m1[1] - m2[1]
        # cast the gradients from gpus to cpus
        tmp = tmp
        
        if q_option in ['dp_both', 'dp_up']:
            gradient_data[m1[0]] = add_dp_noise(tmp.detach(), sigma_u, q_clip)
        elif q_option == 'none':
            gradient_data[m1[0]] = tmp.detach()
        elif q_option == 'cosine':
            quantized_grad, norm_grad, bound_grad = cosine_quantization(tmp, n_bits, q_clip)
            gradient_data[m1[0]] = quantized_grad.detach()
            gradient_rec1[m1[0]] = norm_grad.detach()
            gradient_rec2[m1[0]] = bound_grad.detach()
        elif q_option == 'linear':
            quantized_grad, min_grad, max_grad, diag_grad = linear_quantization(tmp, n_bits, args.quantize_unbiased, args.quantize_hadamard)
            gradient_data[m1[0]] = quantized_grad.detach()
            gradient_rec1[m1[0]] = min_grad.detach()
            gradient_rec2[m1[0]] = max_grad.detach()
            gradient_rec3[m1[0]] = diag_grad.detach()
        else:
            print("Unexpected quantization method:", sys.exc_info()[0])
            raise NotImplementedError(f'Unexpected quantization method {q_option}.')
    
    buffer['gradient_data'].append(gradient_data)
    buffer['gradient_rec1'].append(gradient_rec1)
    buffer['gradient_rec2'].append(gradient_rec2)
    buffer['gradient_rec3'].append(gradient_rec3)

def noniid(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all

def loss_prox(model_global, model_local, device):
    loss = torch.tensor(0.0).to(device)
    for m1, m2 in zip( model_global.named_parameters() , model_local.named_parameters() ):
        assert m1[0] == m2[0]
        assert m1[1].shape == m2[1].shape
        
        loss += torch.norm(m1[1].detach() - m2[1]) ** 2
        
    return loss
