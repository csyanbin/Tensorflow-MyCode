"""
    tensorflow video data reader
    DONE:
        uniformly read same frames from each video
    TODO:
        1. optical flow reader
        2. random read same frames from each video (if not enough)
        3. data preprocessing, such as Ncrops, norm
"""


from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import random

import matplotlib
import matplotlib.pyplot as plt
import os

# fix random seed for testing
import random
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)


def generate_data():
    " generate simulated data for testing "
    for v in range(10):
        if not os.path.exists('video_example/video'+str(v)):
            os.mkdir('video_example/video'+str(v))
        for i in range(1,21):
            _x = np.random.randint(0, 256, size=(10,10,3))
            plt.imsave("video_example/video{}/image_{:04d}.jpg".format(v,i), _x)

def _read_from_disk_spatial(fpath, nframes, num_samples=25, start_frame=0,
                            file_prefix='', file_zero_padding=4, file_index=1,
                            dataset_dir='', step=None):
    duration = nframes
    if step is None:
      if num_samples == 1:
          step = tf.random_uniform([1], 0, nframes, dtype='int32')[0]
      else:
          step = tf.cast((duration-tf.constant(1)) /
                         (tf.constant(num_samples-1)), 'int32')
    allimgs = []
    with tf.variable_scope('read_rgb_video'):
        for i in range(num_samples):
            if num_samples == 1:
                i = 1  # so that the random step value can be used
            with tf.variable_scope('read_rgb_image'):
                prefix = file_prefix + '_' if file_prefix else ''
                impath = tf.string_join([
                    tf.constant(dataset_dir),
                    fpath, tf.constant('/'),
                    prefix,
                    tf.as_string(start_frame + i * step + file_index,
                      width=file_zero_padding, fill='0'),
                    tf.constant('.jpg')])
                img_str = tf.read_file(impath)
                image = tf.image.decode_image(img_str, channels=3)
                # preprocessing image here

            allimgs.append(image)
    return allimgs

def read_my_file_format(input_line, num_samples):
  """
  Args:
    input_line: folder, nframes, label with ' ' seperated

  Returns:
    Two tensors: [nframes,W,H,C] data and [] label.
  """
  fpath, nframes, label = tf.decode_csv(input_line, [[""], [-1], [-1]], " ")
  allimgs =  _read_from_disk_spatial(fpath, nframes, num_samples=num_samples, file_prefix='image')
  allimgs = tf.stack(allimgs, axis=0)
  return allimgs, label, fpath

def prepare_batch_data(list_file, batch_size, num_samples=10, num_epochs=1, num_threads=1):
    input_list = [line.strip() for line in open(list_file).readlines()]
    input_queue = tf.train.string_input_producer(input_list, num_epochs, shuffle=True)
    
    # single reader with multiple threds
    video_imgs, labels, fpath = read_my_file_format(input_queue.dequeue(), num_samples)
    img_batch, label_batch, fname_batch = tf.train.batch([video_imgs, labels, fpath],  
                                           num_threads=num_threads, shapes=([num_samples, 10, 10, 3], [], []), batch_size=batch_size)

    # use multiple reader
    # data_list = [read_my_file_format(input_queue.dequeue(), num_samples)
    #                  for _ in range(num_threads)]
    # img_batch, label_batch, fname_batch = tf.train.batch_join(data_list, shapes=([num_samples, 10, 10, 3], [], []), batch_size=batch_size)


    return img_batch, label_batch, fname_batch


# Parameters for video data reader
batch_size = 10
num_epochs = 20
num_samples = 10
list_file = 'video_list.txt'
num_threads = 1


img_batch, label_batch, fname_batch = prepare_batch_data(list_file, batch_size, num_samples, num_epochs, num_threads=num_threads)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        num = 0
        while not coord.should_stop():
            fname, img = sess.run([fname_batch, img_batch])
            print(fname)
            print(np.shape(img))
            img = np.reshape(img, [10, -1])
            print(np.sum(img,1))
            num += batch_size
            print(num, "samples have been seen")

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
