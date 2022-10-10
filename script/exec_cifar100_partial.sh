#source /p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/default_env/activate.sh
cd ..

#python main.py -cfg settings/resnet18/cifar100-partial.yaml
#python main.py -cfg settings/resnet18/cifar100-partial-restart.yaml
#python main.py -cfg settings/resnet18/cifar100-partial-restart-10.yaml
python main.py -cfg settings/resnet18/cifar100-partial-restart-10-warmup.yaml
