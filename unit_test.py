import models
import argparse
import pdb

import torch
import torchvision.models as torch_models 

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', '--dataset', default='cifar100', type=str)
parser.add_argument('-strategy', '--strategy', default='partial', type=str)
args = parser.parse_args()

resnet = torch_models.resnet18()
print(resnet)
Network = getattr(models, 'resnet18')
model_server = Network(args)


model_server.set_submodel(0)
for p in model_server.trainable_parameters():
    print(p.size())

''' test the progressive strategy
model_server.set_submodel(3)        
submodel = model_server.gen_submodel()

test_x = torch.rand(1, 3, 32, 32)
pdb.set_trace()
#t_out = resnet(test_x)
o_out = submodel(test_x)
'''

#print(f't_out: {t_out.size()}, o_out: {o_out.size()}')
