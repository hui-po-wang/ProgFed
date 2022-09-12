import h5py
import tensorflow as tf

class generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            import ipdb; ipdb.set_trace()
            for im in hf["train_img"]:
                yield im


hdf5_path = "./datasets/fed_cifar100_train.h5"

ds = tf.data.Dataset.from_generator(
    generator(hdf5_path), 
    tf.uint8, 
    tf.TensorShape([427,561,3]))

import ipdb; ipdb.set_trace()
value = ds.make_one_shot_iterator().get_next()

sess = tf.compat.v1.Session()
# Example on how to read elements
while True:
    try:
        data = sess.run(value)
        print(data.shape)
    except tf.errors.OutOfRangeError:
        print('done.')
        break