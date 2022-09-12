import tensorflow as tf
import tensorflow_federated as tff

dataset = tff.simulation.datasets.cifar100

dataset.load_data(
    cache_dir='./'
)

