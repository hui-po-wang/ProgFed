#source /huipo/default_env/activate.sh
cd ..

# basic setting
python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u.yaml

# advanced federated optimization
#python main.py -cfg settings-fed/resnet18/fedprox-cifar100-prog-warmup25-375u.yaml
#python main.py -cfg settings-fed/resnet18/fedadam-cifar100-prog-warmup25-375u.yaml

# ablation study on #stages and #iterations
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S2.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S3.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S5.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S8.yaml 
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S8-4000t.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S2-2000t.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S2-2500t.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S2-1500.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup100-S2.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup200-S2.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S3-4000t.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-S5-4000t.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-4000t.yaml

# compression
#python main.py -cfg settings-fed/convnet/cifar10-prog-500u-l8.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l8.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-s25.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-s10.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l4.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l2.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l8+s25.yaml
#python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u-l8+s10.yaml
