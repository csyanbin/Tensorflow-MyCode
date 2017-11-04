"""
    tensorflow image data reader
    Using tf.train.slice_input_producer
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random


def generate_data():
    """ generate examples images """
    for i in range(100):
        _x = np.random.randint(0, 256, size=(10, 10, 4))
        plt.imsave("example/image_{}.jpg".format(i), _x)
    

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    fname = input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    # example = tf.image.decode_png(file_contents, channels=3)
    example = tf.image.decode_image(file_contents, channels=4)
    return example, label, fname


def prepare_batch_data(list_file, batch_size, num_epochs=1, num_threads=1):
    # split image path and label from list file
    image_list, label_list = read_labeled_image_list(list_file)
    # convert to tensor
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    # wrap image and label as input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=False)
    
    ################# single reader ######################
    ## decode image, label, filename
    # img, label, fname = read_images_from_disk(input_queue)
    ## make single tensor as batch data
    # img_batch, label_batch, fname_batch = tf.train.batch([img, label, fname], num_threads=num_thread, shapes=([10, 10, 4], [], []), batch_size=batch_size)
    
    ################# use multiple reader ##########################
    data_list = [read_images_from_disk(input_queue)
                    for _ in range(num_threads) ]
    img_batch, label_batch, fname_batch = tf.train.batch_join(data_list, shapes=([10, 10, 4], [], []), batch_size=batch_size)

    return img_batch, label_batch, fname_batch

    

# fix random seed for testing
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
tf.reset_default_graph()

# Hyperparams 
batch_size = 10
num_epochs = 1
batch_size = 10
list_file = 'img_list.txt2'
num_threads = 1


img_batch, label_batch, fname_batch = prepare_batch_data(list_file, batch_size, num_epochs=1, num_threads=1)
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
