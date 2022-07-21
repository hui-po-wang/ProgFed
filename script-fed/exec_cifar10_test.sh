#!/bin/bash
#SBATCH --nodes=1 # How many nodes? 
#SBATCH -A hai_tfda_wp_2_2 # Who pays for it?
#SBATCH --partition booster
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00 
#SBATCH -o output-eval.txt 
#SBATCH -e error-eval.txt
# Where does the code run?
# Required for legacy reasons
# How long?

source /p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/default_env/activate.sh
cd ..
srun python test.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u.yaml

