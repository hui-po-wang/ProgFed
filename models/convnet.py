import torch
import torch.nn as nn
import torchvision.models as models

from .model_utils import SingleSubModel, MultiSubModel, get_outdim

subnet_strategies = ['progressive', 'dense', 'mixed']
fullnet_strategies = ['baseline', 'partial', 'layerwise']

class ConvNet(nn.Module):
    def __init__(self, dataset, strategy):
        super(ConvNet, self).__init__()

        self.dataset = dataset
        self.strategy = strategy
        self.num_classes = get_outdim(dataset)

        self.module_splits = nn.ModuleList()
        self.head_splits = []
        self.classifier = None

        self.ind = -1
        self.enc = None
        self.head = None

        if self.dataset == 'cifar10':
            self.classifier = nn.Linear(64, 16)

            # conv1
            self.module_splits.append(nn.Sequential(nn.Conv2d(3, 32, 3),
                                                    nn.ReLU(),
                                                    nn.MaxPool2d((2, 2))))
            # conv2_1
            self.module_splits.append(nn.Sequential(nn.Conv2d(32, 64, 3),
                                                    nn.ReLU(),
                                                    nn.MaxPool2d(2, 2)))
            # conv2_2
            self.module_splits.append(nn.Sequential(nn.Conv2d(64, 64, 3),
                                                    nn.ReLU()))
            # fc1
            self.module_splits.append(nn.Sequential(nn.Flatten(),
                                                    nn.Linear(4*4*64, 64),
                                                    nn.ReLU()))

            # head for conv1 
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 10)))
            # head for conv2_1
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10)))
            # head for conv2_2
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10)))
            # head for fc1
            self.head_splits.append(self.classifier)
        elif self.dataset == 'emnist':
            self.classifier = nn.Linear(128, 62)

            # conv1
            self.module_splits.append(nn.Sequential(nn.Conv2d(1, 32, 3, 1, 0),
                                                    nn.ReLU()))
            # conv2
            self.module_splits.append(nn.Sequential(nn.Conv2d(32, 64, 3, 1, 0),
                                                    nn.ReLU(),
                                                    nn.MaxPool2d((2, 2)),
                                                    nn.Dropout(0.25)))
            # fc1
            self.module_splits.append(nn.Sequential(nn.Flatten(),
                                                    nn.Linear(12*12*64, 128),
                                                    nn.ReLU(),
                                                    nn.Dropout(0.5)))
            # head for conv1
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, self.num_classes)))
            # head for conv2_1
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.num_classes)))
            # head for fc1
            self.head_splits.append(self.classifier)

        elif self.dataset == 'mnist':
            self.classifier = nn.Linear(512, 16)

            # conv1
            self.module_splits.append(nn.Sequential(nn.Conv2d(1, 32, 5, 1, 2),
                                                    nn.ReLU(),
                                                    nn.MaxPool2d((2, 2))))
            # conv2
            self.module_splits.append(nn.Sequential(nn.Conv2d(32, 64, 5, 1, 2),
                                                    nn.ReLU(),
                                                    nn.MaxPool2d((2, 2))))
            # fc1
            self.module_splits.append(nn.Sequential(nn.Flatten(),
                                                    nn.Linear(7*7*64, 512),
                                                    nn.ReLU()))

            # head for conv1
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 10)))
            # head for conv2_1
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10)))
            # head for fc1
            self.head_splits.append(self.classifier)

        else:
            raise NotImplementedError()

    def forward(self, x):
        out = x 
        for m in sef.module_splits:
            out = m(out)
        out = self.classifier(out)
        return out
        '''
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_2(x))
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        '''

    def set_submodel(self, ind, strategy=None):
        # mnist models only has three stages
        if 'mnist' in self.dataset and ind == 3:
            ind = 2
        self.ind = ind

        assert ind <= 3
        if strategy == None:
            strategy = self.strategy
        print(strategy, ind, self.dataset)
        if strategy in subnet_strategies:
            '''progressive, mixed, dense'''
            modules = []
            for i in range(ind+1):
                modules.append(self.module_splits[i])
            self.enc = nn.Sequential(*modules)
            self.head = self.head_splits[ind]

        elif strategy in fullnet_strategies:
            '''baseline, layerwise, partial'''
            modules = []
            for i in range(len(self.module_splits)):
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

def convnet(args):
    return ConvNet(args.dataset, args.strategy)

