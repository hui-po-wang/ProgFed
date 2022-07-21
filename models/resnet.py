'''some code from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
'''
import torch
import torch.nn as nn
import torchvision.models as models

from .model_utils import SingleSubModel, MultiSubModel, get_outdim

subnet_strategies = ['progressive', 'dense', 'mixed']
fullnet_strategies = ['baseline', 'partial', 'layerwise', 'svcca']

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.GroupNorm(2, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            nn.GroupNorm(2, out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(out_channels * BasicBlock.expansion)
                nn.GroupNorm(2, out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.GroupNorm(2, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.GroupNorm(2, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            #nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            nn.GroupNorm(2, out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                #nn.BatchNorm2d(out_channels * BottleNeck.expansion)
                nn.GroupNorm(2, out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, dataset, strategy, num_stages):
        super().__init__()

        self.strategy = strategy
        self.num_stages = num_stages
        self.num_classes = get_outdim(dataset)

        self.in_channels = 64

        if dataset == 'imagenet':
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
                #nn.BatchNorm2d(64),
                nn.GroupNorm(2, 64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))    
        else:
            # This layer differs from the one for imagenet due to the input resolution
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                #nn.BatchNorm2d(64),
                nn.GroupNorm(2, 64),
                nn.ReLU(inplace=True))
        
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        # add to a list, which is prepared for progressive learning
        if self.num_stages == 8:
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1, self.conv2_x[:num_block[0]//2]))
            self.module_splits.append(self.conv2_x[num_block[0]//2:])
            self.module_splits.append(self.conv3_x[:num_block[1]//2])
            self.module_splits.append(self.conv3_x[num_block[1]//2:])
            self.module_splits.append(self.conv4_x[:num_block[2]//2])
            self.module_splits.append(self.conv4_x[num_block[2]//2:])
            self.module_splits.append(self.conv5_x[:num_block[3]//2])
            self.module_splits.append(self.conv5_x[num_block[3]//2:])

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(512 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                              nn.Flatten(),
                                              self.fc))

        if self.num_stages == 5:
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                              self.conv2_x[:num_block[0]//2]))
            self.module_splits.append(nn.Sequential(self.conv2_x[num_block[0]//2:],
                                              self.conv3_x[:num_block[1]//2]))
            self.module_splits.append(nn.Sequential(self.conv3_x[num_block[1]//2:],
                                              self.conv4_x[:num_block[2]//2]))
            self.module_splits.append(nn.Sequential(self.conv4_x[num_block[2]//2:],
                                              self.conv5_x[:num_block[3]//2]))
            self.module_splits.append(self.conv5_x[num_block[3]//2:])

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(512 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                              nn.Flatten(),
                                              self.fc))
        # the default setting in the ProgFed paper
        elif self.num_stages == 4:
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                              self.conv2_x))
            self.module_splits.append(self.conv3_x)
            self.module_splits.append(self.conv4_x)
            self.module_splits.append(self.conv5_x)

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                              nn.Flatten(),
                                              self.fc))
        elif self.num_stages == 3:
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                              self.conv2_x))
            self.module_splits.append(nn.Sequential(self.conv3_x,
                                              self.conv4_x))
            self.module_splits.append(self.conv5_x)

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                              nn.Flatten(),
                                              self.fc))

        elif self.num_stages == 2:
            '''
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                              self.conv2_x,
                                              self.conv3_x))
            self.module_splits.append(nn.Sequential(self.conv4_x,
                                              self.conv5_x))

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                              nn.Flatten(),
                                              self.fc))
            '''
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                              self.conv2_x,
                                              self.conv3_x,
                                              self.conv4_x))
            self.module_splits.append(nn.Sequential(self.conv5_x))

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                              nn.Flatten(),
                                              self.fc))


        self.ind = -1
        self.enc = None
        self.head = None

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = x
        for m in self.module_splits:
            output = m(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def dense_forward(self, x):
        results = []
        out = x
        for i in range(self.num_stages):
            out = self.module_splits[i](out)
            results.append(self.head_splits[i](out))
        return results

    def set_submodel(self, ind, strategy=None):
        self.ind = ind
        assert ind <= self.num_stages - 1
        if strategy == None:
            strategy = self.strategy

        if strategy in subnet_strategies:
            '''progressive, mixed, dense'''
            modules = []
            for i in range(ind+1):
                modules.append(self.module_splits[i])
            self.enc = nn.Sequential(*modules)
            self.head = self.head_splits[ind]

        elif strategy in fullnet_strategies:
            '''baseline, layerwise, partial, svcca'''
            modules = []
            for i in range(self.num_stages):
                modules.append(self.module_splits[i])
            self.enc = nn.Sequential(*modules)
            self.head = nn.Sequential(self.avg_pool,
                                      nn.Flatten(),
                                      self.fc)
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


def resnet18(args):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], args.dataset, args.strategy, args.num_stages)

def resnet34(args):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], args.dataset, args.strategy, args.num_stages)

def resnet50(args):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], args.dataset, args.strategy, args.num_stages)

def resnet101(args):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], args.dataset, args.strategy, args.num_stages)

def resnet152(args):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], args.dataset, args.strategy, args.num_stages)

if __name__ == "__main__":
    model_server = ResNet18()
    '''
    x = torch.randn(1, 3, 224, 224)

    for i in range(4):
        model_server.set_submodel(i)
        submodel = model_server.gen_submodel()
        
        pred = submodel(x)
        print(pred.shape)
    '''
    model_server.set_submodel(0)
    for p in model_server.trainable_parameters():
        print(p)
