from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
tf.reset_default_graph()

# Hyperparams
batch_size = 10
num_epochs = 1

for i in range(100):
    _x = np.random.randint(0, 256, size=(10, 10, 4))
    plt.imsave("example/image_{}.jpg".format(i), _x)

images = tf.train.match_filenames_once('example/*.jpg')

fname_q = tf.train.string_input_producer(images, num_epochs=num_epochs, shuffle=True)

def read_data(fname_q):
    reader = tf.WholeFileReader()
    fname, value = reader.read(fname_q)
    print(fname,value)
    img = tf.image.decode_image(value)

    return img, fname

# use single reader 
# img, fname = read_data(fname_q)
# img_batch, fname_batch = tf.train.batch([img, fname], num_threads=3, shapes=([10, 10, 4], []), batch_size=batch_size)

# use multiple reader
img_list = [read_data(fname_q)
                for _ in range(2) ]
img_batch, fname_batch = tf.train.batch_join(img_list, shapes=([10, 10, 4], []), batch_size=batch_size)


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_samples = 0
    try:
        while not coord.should_stop():
            fname, img = sess.run([fname_batch, img_batch])
            print(fname)
            img = np.reshape(img, [10, -1])
            print(np.sum(img,1))
            num_samples += batch_size
            print(num_samples, "samples have been seen")

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
