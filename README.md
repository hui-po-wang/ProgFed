# ProgFed: Effective, Communication, and Computation Efficient Federated Learning by Progressive Training

#### [[Paper (ICML 2022)]](https://arxiv.org/abs/2110.05323)
------------------------------------

### Citation
```
@inproceedings{wang2022progfed,
  title={ProgFed: Effective, Communication, and Computation Efficient Federated Learning by Progressive Training},
  author={Wang, Hui-Po and Stich, Sebastian and He, Yang and Fritz, Mario},
  booktitle={International Conference on Machine Learning},
  pages={23034--23054},
  year={2022},
  organization={PMLR}
}
```
------------------------------------
## Quick Start
To quickly reproduce the main result of our work, follow the instructions below.

- Modify the ```--data_path``` argument in ```main.py``` accordingly (or specify it in the scripts)
- Run scripts
```
cd script-fed
# run end-to-end baselines
bash exec_baseline.sh
# run ProgFed
bash exec_prog.sh
```
------------------------------------
## Installation
- Pytorch >= 1.8.0
------------------------------------
## Reproduce Results
In ```script-fed/```, we provide ```exec_baseline.sh``` and ```exec_fed.sh``` for reproducing the experiments with an end-to-end baseline model and with our progressive training algorithm. We also offer detailed usage as follows.

### Basci Usage
```
python main.py -cfg <path-to-configuration-file>
```
``` settings-fed``` contains various configurations. They are categorized by architectures, e.g., ```convnet```, ```resnet18```, and ```vgg16```.
### ProgFed vs. End-to-End Baseline
```
# ProgFed
python main.py -cfg settings-fed/resnet18/cifar100-prog-warmup25-375u.yaml
# end-to-end baseline
python main.py -cfg settings-fed/resnet18/cifar100-baseline.yaml
```
### Compression
We show that ProgFed is compatible with sparsification and quantization. The configuration files specify the number of bits for linear quantization and sparsification ratio, e.g., ```cifar100-baseline-l8+s25.yaml``` stands for ```8-bit``` linear quantization along with ```25%``` sparsification.
- ProgFed
```
├── settings-fed/resnet18
    ├── cifar100-prog-warmup25-375u-l8.yaml
    ├── cifar100-prog-warmup25-375u-l4.yaml
    ├── cifar100-prog-warmup25-375u-l2.yaml
    ├── cifar100-prog-warmup25-375u-s25.yaml
    ├── cifar100-prog-warmup25-375u-s10.yaml
    ├── cifar100-prog-warmup25-375u-l8+s25.yaml
    └── cifar100-prog-warmup25-375u-l8+s10.yaml

```
- Baseline
```
├── settings-fed/resnet18
    ├── cifar100-baseline-s25.yaml
    ├── cifar100-baseline-s10.yaml
    ├── cifar100-baseline-l2.yaml
    ├── cifar100-baseline-l4.yaml
    ├── cifar100-baseline-l8.yaml
    ├── cifar100-baseline-l8+s10.yaml
    └── cifar100-baseline-l8+s25.yaml

```
