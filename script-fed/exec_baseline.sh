#source /huipo/default_env/activate.sh
cd ..

# basic settings
#python main.py -cfg settings-fed/convnet/cifar10-baseline.yaml
#python main.py -cfg settings-fed/resnet18/cifar10-baseline.yaml
#python main.py -cfg settings-fed/vgg16/cifar10-baseline.yaml
python main.py -cfg settings-fed/resnet18/cifar100-baseline.yaml
#python main.py -cfg settings-fed/convnet/emnist-baseline.yaml
#python main.py -cfg settings-fed/resnet18/fedprox-cifar100-baseline.yaml
#python main.py -cfg settings-fed/resnet18/fedadam-cifar100-baseline.yaml 
#python main.py -cfg settings-fed/convnet/fedadam-emnist-baseline.yaml


# compression
#python main.py -cfg settings-fed/convnet/cifar10-baseline-l8.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-baseline-l8.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-baseline-s25.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-baseline-s10.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-baseline-l4.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-baseline-l2.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-baseline-l8+s10.yaml -seed 123
#python main.py -cfg settings-fed/resnet18/cifar100-baseline-l8+s25.yaml -seed 123
