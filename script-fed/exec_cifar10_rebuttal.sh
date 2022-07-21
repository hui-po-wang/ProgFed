#!/bin/bash
#SBATCH --nodes=1 # How many nodes? 
#SBATCH -A hai_tfda_wp_2_2 # Who pays for it?
#SBATCH --partition booster 
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00 
#SBATCH -o output-rebuttal-3.txt 
#SBATCH -e error-rebuttal-3.txt
# Where does the code run?
# Required for legacy reasons
# How long?

source /p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/default_env/activate.sh
cd ..

#srun python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u.yaml
#srun python main.py -cfg settings-fed/rebuttal/fedprox-cifar100-prog-1.yaml
#srun python main.py -cfg settings-fed/rebuttal/fedprox-cifar100-prog-1e-1.yaml
srun python main.py -cfg settings-fed/rebuttal/fedprox-cifar100-prog-1e-2.yaml
