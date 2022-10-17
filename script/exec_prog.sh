#source /p/home/jusers/wang34/juwels/hai_tfda_wp_2_2/huipo/default_env/activate.sh
cd ..

python main.py -cfg settings/resnet18/cifar100-prog.yaml
#python main.py -cfg settings/resnet18/cifar100-prog-restart.yaml
#python main.py -cfg settings/resnet18/cifar100-prog-restart-10.yaml
#python main.py -cfg settings/resnet18/cifar100-prog-restart-10-warmup.yaml
#python main.py -cfg settings/resnet152/cifar100-prog-restart-10-warmup.yaml
#python main.py -cfg settings/resnet152/cifar100-prog-restart-10-warmup-dyn.yaml
#python main.py -cfg settings/resnet152/cifar100-prog-restart-10-warmup-idyn.yaml
#python main.py -cfg settings/vgg16/cifar100-prog-restart-10-warmup.yaml
#python main.py -cfg settings/vgg19/cifar100-prog-restart-10-warmup.yaml
#python main.py -cfg settings/vgg16/cifar100-prog-restart-10-warmup.yaml
#python main.py -cfg settings/vgg19/cifar100-prog-restart-10.yaml
#python main.py -cfg settings/convnet/cifar10-prog-restart-10-warmup.yaml
