#!/bin/bash
#SBATCH --nodes=1 # How many nodes? 
#SBATCH -A hai_tfda_wp_2_2 # Who pays for it?
#SBATCH --partition booster 
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00 
#SBATCH -o output-baseline-13.txt 
#SBATCH -e error-baseline-13.txt
# Where does the code run?
# Required for legacy reasons
# How long?

source /p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/default_env/activate.sh
cd ..
#srun python main.py -cfg settings-fed/convnet/cifar10-baseline.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar10-baseline.yaml
#srun python main.py -cfg settings-fed/vgg16/cifar10-baseline.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-baseline.yaml
#srun python main.py -cfg settings-fed/convnet/emnist-baseline.yaml
#srun python main.py -cfg settings-fed/resnet18/fedprox-cifar100-baseline.yaml -seed 456
#srun python main.py -cfg settings-fed/resnet18/scaffold-cifar100-baseline.yaml
#srun python main.py -cfg settings-fed/resnet18/fedadam-cifar100-baseline.yaml -seed 456
#srun python main.py -cfg settings-fed/resnet18/fedadam-cifar100-baseline-wocmom.yaml
#srun python main.py -cfg settings-fed/convnet/fedadam-emnist-baseline.yaml
# 10 ^
#srun python main.py -cfg settings-fed/resnet18/dpboth-cifar100-baseline.yaml
#srun python main.py -cfg settings-fed/resnet18/dpup-cifar100-baseline.yaml
srun python main.py -cfg settings-fed/resnet18/cifar100-baseline-4000t.yaml

# compression
#srun python main.py -cfg settings-fed/convnet/cifar10-baseline-l8.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-baseline-l8.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-baseline-s25.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-baseline-s10.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-baseline-l4.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-baseline-l2.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-baseline-l8+s10.yaml -seed 123
#srun python main.py -cfg settings-fed/resnet18/cifar100-baseline-l8+s25.yaml -seed 123
