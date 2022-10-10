"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import torchvision.models as models

from .model_utils import SingleSubModel, MultiSubModel, get_outdim

subnet_strategies = ['progressive', 'dense', 'mixed']
fullnet_strategies = ['baseline', 'partial', 'layerwise']

cfg = {
    'A' : [[64,     'M', 128,      'M'], [256, 256,           'M'], [512, 512,           'M'], [512, 512,           'M']],
    'B' : [[64, 64, 'M', 128, 128, 'M'], [256, 256,           'M'], [512, 512,           'M'], [512, 512,           'M']],
    'D' : [[64, 64, 'M', 128, 128, 'M'], [256, 256, 256,      'M'], [512, 512, 512,      'M'], [512, 512, 512,      'M']],
    'E' : [[64, 64, 'M', 128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']]
}

class VGG(nn.Module):

    def __init__(self, module_splits, dataset, strategy):
        super().__init__()
        self.strategy = strategy
        self.num_classes = get_outdim(dataset)

        self.module_splits = module_splits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes))

        self.head_splits = []
        self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Flatten(),
                                          nn.Linear(128, self.num_classes)))
        self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Flatten(),
                                          nn.Linear(256, self.num_classes)))
        self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Flatten(),
                                          nn.Linear(512, self.num_classes)))
        self.head_splits.append(self.classifier)

        self.ind = -1
        self.enc = None
        self.head = None

    def forward(self, x):
        out = x
        for m in self.module_splits:
            out = m(out)
        out = self.classifier(out)

        return out

    def set_submodel(self, ind, strategy=None):
        self.ind = ind
        assert ind <= 3
        if strategy == None:
            strategy = self.strategy

        if strategy in subnet_strategies:
            '''progressive, mixed, dense'''
            if ind == 0:
                modules = []
                for i in range(ind+1):
                    modules.append(self.module_splits[i])
                self.enc = nn.Sequential(*modules)
                self.head = self.head_splits[ind]

            elif ind == 1:
                modules = []
                for i in range(ind+1):
                    modules.append(self.module_splits[i])
                self.enc = nn.Sequential(*modules)
                self.head = self.head_splits[ind]

            elif ind == 2:
                modules = []
                for i in range(ind+1):
                    modules.append(self.module_splits[i])
                self.enc = nn.Sequential(*modules)
                self.head = self.head_splits[ind]

            elif ind == 3:
                modules = []
                for i in range(ind+1):
                    modules.append(self.module_splits[i])
                self.enc = nn.Sequential(*modules)
                self.head = self.head_splits[ind]

        elif strategy in fullnet_strategies:
            '''baseline, layerwise, partial'''
            modules = []
            for i in range(4):
                modules.append(self.module_splits[i])
            self.enc = nn.Sequential(*modules)
            self.head = self.classifier
        else:
            raise NotImplementedError()
    
    def gen_submodel(self):
        if self.strategy == 'dense':
            return MultiSubModel(self.enc, self.head_splits[:self.ind+1], self.strategy, self.ind)
        else:
            return SingleSubModel(self.enc, self.head, self.strategy, self.ind)

    def return_stage_parameters(self):
        out = []
        for i in range(len(self.module_splits)):
            num = 0 
            for p in self.module_splits[i].parameters():
                num += torch.numel(p)
            for p in self.head_splits[i].parameters():
                num += torch.numel(p)
            out.append(num)
        return out 

    def return_num_parameters(self):
        total = 0
        for p in self.trainable_parameters():
            total += torch.numel(p)

        return total

def make_layers(cfg, batch_norm=False, dropout=False):
    models = nn.ModuleList()

    input_channel = 3
    for c in cfg:
        layers = []
        for l in c:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]

            if dropout:
                layers += [nn.Dropout2d(0.1)]
            input_channel = l
        models.append(nn.Sequential(*layers))

    return models

def vgg11_bn(args):
    return VGG(make_layers(cfg['A'], batch_norm=True), args.dataset, args.strategy)

def vgg13_bn(args):
    return VGG(make_layers(cfg['B'], batch_norm=True), args.dataset, args.strategy)

def vgg16_bn(args):
    return VGG(make_layers(cfg['D'], batch_norm=True), args.dataset, args.strategy)

def vgg19_bn(args):
    return VGG(make_layers(cfg['E'], batch_norm=True), args.dataset, args.strategy)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unit test for VGG')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--strategy', default=None)

    args = parser.parse_args()
    model = vgg13_bn(args)
    x = torch.randn(1, 3, 32, 32)
    out = x
    for m in model.module_splits:
        out = m(out)
        print(out.size())


