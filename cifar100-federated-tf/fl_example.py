#import nest_asyncio
#nest_asyncio.apply()

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

import os

from PIL import Image

np.random.seed(0)

tff.federated_computation(lambda: 'Hello, World!')()

emnist_train, emnist_test = tff.simulation.datasets.cifar100.load_data()


print(len(emnist_train.client_ids))

emnist_train.element_type_structure

for idx in range(500):
  if not os.path.exists("./cifar100_fl/%03d"%idx):
    os.mkdir("./cifar100_fl/%03d"%idx)

  example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[idx])

  labels = []

  id_ex = 0
  for example_element in example_dataset.take(100):
    data = example_element['image'].numpy()
    labels.append( example_element['label'].numpy() )

    new_im = Image.fromarray(data)
    new_im.save( "./cifar100_fl/%03d/%03d.png"%(idx,id_ex) )
    np.save("./cifar100_fl/%03d/gt.npy"%idx, np.array(labels))
    id_ex += 1