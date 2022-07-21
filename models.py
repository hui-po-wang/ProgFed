# import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_MNIST(nn.Module):
    def __init__(self):
        super(Net_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 16)

        self.head = None
        self.cur_model = None

    def forward(self, x):
        assert self.head is not None
        assert self.cur_model is not None

        out = self.cur_model(x)
        out = self.head(out)

        return out
        '''
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
        '''
    def gen_submodel(self):
        model = SubModel(self.cur_model, self.head)
        return model
        
    def set_submodel(self, ind):
        if ind == -1:
            self.head = self.fc2
            self.cur_model = nn.Sequential(self.conv1,
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                self.conv2,
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                self.fc1,
                nn.ReLU())

'''
class Net_CIFAR10(nn.Module):
    def __init__(self):
        super(Net_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2_1 = nn.Conv2d(32, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(4*4*64, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_2(x))
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
'''
class SubModel(nn.Module):
    def __init__(self, cur_model, head):
        super(SubModel, self).__init__()
        self.cur_model = cur_model
        self.head = head

    def forward(self, x):
        x = self.cur_model(x)
        x = self.head(x)
        return x

    def print_weight(self):
        for n, p in self.cur_model.named_parameters():
            print(n, p)

    def return_num_parameters(self):
        total = 0
        for p in self.cur_model.parameters():
            total += torch.numel(p)
        for p in self.head.parameters():
            total += torch.numel(p)
        return total

class Net_CIFAR10(nn.Module):
    def __init__(self):
        super(Net_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2_1 = nn.Conv2d(32, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(4*4*64, 64)
        self.fc2 = nn.Linear(64, 16)

        self.head = None
        self.cur_model = None

    def forward(self, x):
        assert self.head is not None
        assert self.cur_model is not None

        out = self.cur_model(x)
        out = self.head(out)

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
    
    def gen_submodel(self):
        model = SubModel(self.cur_model, self.head)
        return model
        
    def set_submodel(self, ind):
        if ind == -1:
            self.head = self.fc2
            self.cur_model = nn.Sequential(self.conv1,
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                self.conv2_1,
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                self.conv2_2,
                nn.ReLU(),
                nn.Flatten(),
                self.fc1,
                nn.ReLU())
        elif ind == 0:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 10))
            self.cur_model = nn.Sequential(self.conv1,
                nn.ReLU(),
                nn.MaxPool2d((2, 2)))
        elif ind == 1:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10))
            self.cur_model = nn.Sequential(*self.cur_model,
                self.conv2_1,
                nn.ReLU(),
                nn.MaxPool2d(2, 2))
        elif ind == 2:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10))
            self.cur_model = nn.Sequential(*self.cur_model,
                self.conv2_2,
                nn.ReLU())
        elif ind == 3:
            self.head = self.fc2
            self.cur_model = nn.Sequential(*self.cur_model,
                nn.Flatten(),
                self.fc1,
                nn.ReLU())

    def return_num_parameters(self):
        total = 0
        for p in self.cur_model.parameters():
            total += torch.numel(p)
        for p in self.head.parameters():
            total += torch.numel(p)
        return total
