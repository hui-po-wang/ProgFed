#source /p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/default_env/activate.sh
cd ..

python main.py -cfg settings/resnet18/cifar100-mixed.yaml
#python main.py -cfg settings/resnet18/cifar100-mixed-restart-10.yaml
#python main.py -cfg settings/resnet18/cifar100-mixed-restart-10-warmup.yaml
