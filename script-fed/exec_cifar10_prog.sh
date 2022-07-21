#!/bin/bash
#SBATCH --nodes=1 # How many nodes? 
#SBATCH -A hai_tfda_wp_2_2 # Who pays for it?
#SBATCH --partition booster 
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00 
#SBATCH -o output-prog-50.txt 
#SBATCH -e error-prog-50.txt
# Where does the code run?
# Required for legacy reasons
# How long?

source /p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/default_env/activate.sh
cd ..
#srun python main.py -cfg settings-fed/convnet/cifar10-prog.yaml
#srun python main.py -cfg settings-fed/convnet/cifar10-prog-200u.yaml
#srun python main.py -cfg settings-fed/convnet/cifar10-prog-500u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar10-prog-500u.yaml
#srun python main.py -cfg settings-fed/vgg16/cifar10-prog-500u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-750u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-restart-750u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-restart-warmup-750u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar10-prog-restart-500u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar10-prog-restart-warmup-500u.yaml
# 10 ^
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup-750u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-100u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup5-750u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup100-750u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u.yaml
#srun python main.py -cfg settings-fed/convnet/cifar10-prog-warmup-250u.yaml
#srun python main.py -cfg settings-fed/convnet/cifar10-prog-warmup10-250u.yaml
#srun python main.py -cfg settings-fed/convnet/emnist-prog-500u.yaml
#srun python main.py -cfg settings-fed/convnet/emnist-prog-warmup5-500u.yaml
#srun python main.py -cfg settings-fed/convnet/emnist-prog-warmup25-500u.yaml
# 20 ^
#srun python main.py -cfg settings-fed/convnet/emnist-prog-warmup5-300u.yaml
#srun python main.py -cfg settings-fed/convnet/emnist-prog-warmup5-400u.yaml
#srun python main.py -cfg settings-fed/convnet/emnist-prog-warmup5-250u.yaml
#srun python main.py -cfg settings-fed/convnet/cifar10-prog-250u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-500u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-625u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-750u.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-375u-svcca.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-375u.yaml
# 29 ^
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S2.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S3.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S5.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S8.yaml 
#srun python main.py -cfg settings-fed/resnet18/fedprox-cifar100-prog-warmup25-375u.yaml
#srun python main.py -cfg settings-fed/resnet18/scaffold-cifar100-prog-warmup25-375u.yaml
#srun python main.py -cfg settings-fed/resnet18/fedprox-cifar100-prog-warmup25-375u-nom.yaml
#srun python main.py -cfg settings-fed/resnet18/fedprox-cifar100-prog-warmup25-375u-piecewise.yaml
#srun python main.py -cfg settings-fed/resnet18/dpup-cifar100-prog-warmup25-375u.yaml
#srun python main.py -cfg settings-fed/resnet18/dpboth-cifar100-prog-warmup25-375u.yaml
#srun python main.py -cfg settings-fed/resnet18/fedadam-cifar100-prog-warmup25-375u.yaml --seed 456
# 40 ^
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S8-4000t.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S2-2000t.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S2-2500t.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S2-1500.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup100-S2.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup200-S2.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S3-4000t.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S5-4000t.yaml
#srun python main.py -cfg settings-fed/convnet/fedadam-emnist-prog-warmup5-250u.yaml
srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-4000t.yaml
# 50 ^

# compression
#srun python main.py -cfg settings-fed/convnet/cifar10-prog-500u-l8.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l8.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-s25.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-s10.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l4.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l2.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l8+s25.yaml
#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l8+s10.yaml
