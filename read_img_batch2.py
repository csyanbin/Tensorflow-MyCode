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

list_file = 'img_list.txt2'

def generate_data():
    """ generate examples images """
    for i in range(100):
        _x = np.random.randint(0, 256, size=(10, 10, 4))
        plt.imsave("example/image_{}.jpg".format(i), _x)


def read_my_file_format(filename_and_label_tensor):
  """Consumes a single filename and label as a ' '-delimited string.

  Args:
    filename_and_label_tensor: A scalar string tensor.

  Returns:
    Two tensors: the decoded image, and the string label.
  """
  filename, label = tf.decode_csv(filename_and_label_tensor, [[""], [""]], " ")
  file_contents = tf.read_file(filename)
  example = tf.image.decode_png(file_contents)
  return example, label, filename    

input_list = [line.strip() for line in open(list_file).readlines()]
input_queue = tf.train.string_input_producer(input_list, num_epochs=num_epochs, shuffle=False)


################# single reader ######################
img, label, fname = read_my_file_format(input_queue.dequeue())
img_batch, label_batch, fname_batch = tf.train.batch([img, label, fname], num_threads=3, shapes=([10, 10, 4], [], []), batch_size=batch_size)

################# use multiple reader ##########################
# data_list = [read_images_from_disk(input_queue)
#                 for _ in range(2) ]
# img_batch, label_batch, fname_batch = tf.train.batch_join(data_list, shapes=([10, 10, 4], [], []), batch_size=batch_size)

batch_size = 10
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
