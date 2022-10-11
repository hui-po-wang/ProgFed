#source /huipo/default_env/activate.sh
cd ..

# basic setting
#python main.py -cfg settings-fed/convnet/cifar10-cen-baseline.yaml
python main.py -cfg settings-fed/convnet/cifar10-cen-prog.yaml
