import argparse
import torch
import numpy as np

from scipy.special import softmax

choice_dict = {}
choice_dict['strategy'] = ['baseline', 'progressive', 'partial', 'layerwise', 'mixed', 'dense', 'random']
choice_dict['dataset'] = ['cifar100', 'cifar10', 'mnist', 'imagenet']

default_dict =  {}
default_dict['warmup'] = False
default_dict['num_stages'] = 4
default_dict['update_strategy'] = None

class Parser(dict):
    def __init__(self, *args):
        super(Parser, self).__init__()
        for d in args:
            if isinstance(d, argparse.Namespace):
                d = vars(d)
            for k, v in d.items():
                if k == 'seed' and k in self.keys() and self[k] != None:
                    print(f'{k} is found in arg parser.')
                    continue
                assert k not in self.keys() or k == 'seed'
                k = k.replace('-', '_')
                #check whether arguments match the limited choices
                if k in choice_dict.keys() and v not in choice_dict[k]:
                    raise ValueError(f'Illegal argument \'{k}\' for choices {choice_dict[k]}')
                self[k] = v

        # check whether the default options has been in args; otherswise, add it.
        for k in default_dict.keys():
            if k not in self.keys():
                self[k] = default_dict[k] 

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, val):
        self[key] = val

class Statistics(object):
    def __init__(self):
        self.dict = {}

    def add(self, key, val):
        if key not in self.dict.keys():
            self.dict[key] = []
        self.dict[key].append(val)

    def avg(self, key):
        return np.sum(self.dict[key])

class UpdateScheduler(object):
    def __init__(self, update_cycles, num_stages=4, update_strategy=None):
        self.update_cycles = update_cycles
        self.num_stages = num_stages
        self.update_strategy = update_strategy
        if isinstance(update_cycles, int):
            self.update_cycles = [update_cycles for _ in range(num_stages-1)]

        if self.update_strategy == 'dynamic':
            self.normalize()
        elif self.update_strategy == 'i_dynamic':
            self.normalize(inverse=True)
        elif self.update_strategy == None:
            pass
        else:
            raise NotImplementedError()

        self.accumulate()

    def __getitem__(self, index):
        assert index < self.num_stages
        return self.update_cycles[index]

    def __str__(self):
        return f'update_cycles: {self.update_cycles}; update_strategy: {self.update_strategy}'

    def accumulate(self):
        for i in range(1, len(self.update_cycles)):
            self.update_cycles[i] += self.update_cycles[i-1]
        self.update_cycles = np.append(self.update_cycles, [1e9])

    def normalize(self, total=75, inverse=False):
        # discard the last one since it will become end-to-end.
        self.update_cycles = np.asarray(self.update_cycles[:-1]) / np.sum(self.update_cycles)
        if inverse:
            self.update_cycles *= -1
        print(self.update_cycles)
        sum_weight = softmax(self.update_cycles)
        self.update_cycles = np.round(sum_weight * total)
        print(self.update_cycles)


class LearningScheduler(object):
    def __init__(self, **kwargs):
        self.type = kwargs['type']
        dummy_opt = torch.optim.SGD(torch.nn.Linear(1,1).parameters(), lr=kwargs['lr'])
        if self.type == 'multistep':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                dummy_opt, milestones=kwargs['milestones'], gamma=kwargs['gamma'])
        elif self.type == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(dummy_opt, 
                T_0=kwargs['T_0'], T_mult=kwargs['T_mult'], eta_min=kwargs['eta_min'])
        else:
            raise NotImplementedError()

        self.opt = None
        self.step_cnt = 0
        self.milestone = kwargs['milestones']

    def set_opt(self, opt):
        self.opt = opt
        lr = self.get_lr()
        for g in self.opt.param_groups:
            g['lr'] = lr

    def step(self):
        assert self.opt is not None
        
        self.step_cnt +=1
        '''
        if self.type == 'cosine' and self.step_cnt in self.milestone:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(dummy_opt, 
                T_0=kwargs['T_0'], T_mult=kwargs['T_mult'], eta_min=kwargs['eta_min'])
        '''

        self.lr_scheduler.step()
        for g in self.opt.param_groups:
            g['lr'] = self.get_lr()

    def get_lr(self):
        return self.lr_scheduler.get_last_lr()[0]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
   import yaml 
