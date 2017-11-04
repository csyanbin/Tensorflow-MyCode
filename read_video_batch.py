"""
    tensorflow video data reader
    DONE:
        uniformly read same frames from each video
    TODO:
        0. check label - and no shuffle
        1. random read same frames from each video (if not enough, rand fill)
        2. read img+flow
        3. new Dataset API
        4. apply BGR flip if use caffe pretrained models
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt
import os

from preprocessing.preprocessing_factory import get_preprocessing

IM_WD = 340
IM_HT = 256

_RE_W = 224
_RE_H = 224

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean(
        'is_training', True, 'train or eval step')
tf.app.flags.DEFINE_integer(
        'ncrops', 1, 'default 1 for train and 5 for testing')
tf.app.flags.DEFINE_string(
        'pre_type', 'all', 'single or all video preprocess')


def generate_data():
    " generate simulated data for testing "
    for v in range(10):
        if not os.path.exists('video_example/frames/class'+str(v)):
            os.system('mkdir -p video_example/frames/class'+str(v))
        for i in range(1,21):
            _x = np.random.randint(0, 256, size=(IM_HT, IM_WD, 3))
            plt.imsave("video_example/frames/class{}/image_{:04d}.jpg".format(v,i), _x)

def generate_data_flow():
    " generate simulated optical flow data for testing "
    for v in range(10):
        if not os.path.exists('video_example/flow/class'+str(v)):
            os.system('mkdir -p video_example/flow/class'+str(v))
        for i in range(1,21):
            _x = np.random.randint(0, 256, size=(IM_HT, IM_WD))
            _y = np.random.randint(0, 256, size=(IM_HT, IM_WD))
            plt.imsave("video_example/flow/class{}/flow_x_{:04d}.jpg".format(v,i), _x, cmap=plt.cm.gray)
            plt.imsave("video_example/flow/class{}/flow_y_{:04d}.jpg".format(v,i), _y, cmap=plt.cm.gray)




def _read_from_disk_spatial(fpath, nframes, num_samples=25, start_frame=0,
                            file_prefix='', file_zero_padding=5, file_index=1,
                            dataset_dir='', step=None):
    " Read video frames from specified folder"
    duration = nframes
    if step is None:
      if num_samples == 1:
          step = tf.random_uniform([1], 0, nframes, dtype='int32')[0]
      else:
          step = tf.cast((duration-tf.constant(1)) /
                         (tf.constant(num_samples-1)), 'int32')
    allimgs = []
    dataset_dir += '/frames/'
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
                
                # read and resize
                img_str = tf.read_file(impath)
                image = tf.image.decode_jpeg(img_str, channels=3)
                ## convert to float32 in range(0,1)
                image = tf.image.convert_image_dtype(image, tf.float32)
                image = tf.image.resize_images(image, [IM_HT, IM_WD])
                # preprocessing image here
                if FLAGS.pre_type=='single':
                    preprocess_fn = get_preprocessing('vgg_ucf', is_training=FLAGS.is_training)
                    image = preprocess_fn(image, _RE_H, _RE_W, ncrops=FLAGS.ncrops)
                    # [1*224*224*3] for 1crop and [5*224*224*3] for 5crop
                else:
                    image = tf.expand_dims(image,0) 

            allimgs.append(image)

        allimgs = tf.stack(allimgs, 0)
        allimgs = tf.reshape(allimgs, [-1]+allimgs.shape.as_list()[2:])
        if FLAGS.pre_type=="all":
            preprocess_fn = get_preprocessing('vgg_ucf', is_training=FLAGS.is_training)
            allimgs = preprocess_fn(allimgs, _RE_H, _RE_W, ncrops=FLAGS.ncrops)

        # sample*ncrops x 224 x 224 x 3

    return allimgs

def _read_from_disk_temporal(fpath, nframes, num_samples=25,
                             optical_flow_frames=10, start_frame=0,
                             file_prefix='', file_zero_padding=5, file_index=1,
                             dataset_dir='', step=None):
    " Read video flows from specified folder "
    duration = nframes
    if step is None:
      if num_samples == 1:
          step = tf.random_uniform([1], 0, nframes-optical_flow_frames-1, dtype='int32')[0]
      else:
          step = tf.cast((duration-tf.constant(optical_flow_frames)) /
                         (tf.constant(num_samples)), 'int32')
    allimgs = []
    dataset_dir += '/flow/'
    with tf.variable_scope('read_flow_video'):
        for i in range(num_samples):
            if num_samples == 1:
                i = 1  # so that the random step value can be used
            with tf.variable_scope('read_flow_image'):
              flow_img = []
              for j in range(optical_flow_frames):
                with tf.variable_scope('read_flow_channels'):
                  for dr in ['x', 'y']:
                    prefix = file_prefix + '_' if file_prefix else ''
                    impath = tf.string_join([
                        tf.constant(dataset_dir),
                        fpath, tf.constant('/'),
                        prefix, '%s_' % dr,
                        tf.as_string(start_frame + i * step + file_index + j,
                          width=file_zero_padding, fill='0'),
                        tf.constant('.jpg')])
                    
                    # read and resize
                    img_str = tf.read_file(impath)
                    image = tf.image.decode_jpeg(img_str, channels=1)
                    ## convert to float32 in range(0,1)
                    image = tf.image.convert_image_dtype(image, tf.float32)
                    image = tf.image.resize_images(image, [IM_HT, IM_WD])
                    # preprocessing here
                    if FLAGS.pre_type=='single':
                        preprocess_fn = get_preprocessing('vgg_ucf', is_training=FLAGS.is_training)
                        image = preprocess_fn(image, _RE_H, _RE_W, ncrops=FLAGS.ncrops)
                    else:
                        image = tf.expand_dims(image,0) 

                    flow_img.append(image) # ncrops x 224 x 224 x 1

              flow_imgs = tf.concat(flow_img, -1) # ncrops x 224 x 224 x 20

              allimgs.append(flow_imgs) # 

        allimgs = tf.stack(allimgs, 0)
        allimgs = tf.reshape(allimgs, [-1]+allimgs.shape.as_list()[2:])
        print(allimgs.shape)
        if FLAGS.pre_type=="all":
            preprocess_fn = get_preprocessing('vgg_ucf', is_training=FLAGS.is_training)
            allimgs = preprocess_fn(allimgs, _RE_H, _RE_W, ncrops=FLAGS.ncrops)

    return allimgs

def read_my_file_format(input_line, num_samples, modality, dataset_dir):
  """
  Args:
    input_line: folder, nframes, label with ' ' seperated
    num_samples: read n frames from video
    dataset_dir: parent folder of flow/frames
  Returns:
    Two tensors: [nframes,W,H,C] data and [] label.
  """
  fpath, nframes, label = tf.decode_csv(input_line, [[""], [-1], [-1]], " ")
  
  if modality=="rgb":
        allimgs =  _read_from_disk_spatial(fpath, nframes, num_samples=num_samples, file_prefix='image', dataset_dir=dataset_dir)
  elif modality.startswith("flow"):
        n = modality[4:]
        optical_flow_frames = int(n)
        allimgs =  _read_from_disk_temporal(fpath, nframes, num_samples=num_samples, file_prefix='flow', dataset_dir=dataset_dir, optical_flow_frames=optical_flow_frames)
  elif modality.startswith("rgb+flow"):
        allimgs = []
        # TODO

  return allimgs, label, fpath


def prepare_batch_data(list_file, batch_size, num_samples, modality="rgb", 
                       dataset_dir="video_example",  num_epochs=1, num_threads=1):

    input_list = [line.strip() for line in open(list_file).readlines()]
    shuffle = 1-FLAGS.is_training
    input_queue = tf.train.string_input_producer(input_list, num_epochs, shuffle=shuffle)
    
    # single reader with multiple threds
    video_imgs, labels, fpath = read_my_file_format(input_queue.dequeue(), num_samples, modality, dataset_dir)
    if modality=="rgb":
        channel = 3
    elif modality.startswith("flow"):
        n = modality[4:]
        channel = 2*int(n)
    elif modality.startswith("rgb+flow"):
        n = modality[modality.rfind('flow')+4:]
        channel = 2*int(n)+3
    
    img_batch, label_batch, fname_batch = tf.train.batch([video_imgs, labels, fpath],  
                                           num_threads=num_threads, shapes=([num_samples*FLAGS.ncrops, _RE_H, _RE_W, channel], [], []), batch_size=batch_size)

    # use multiple reader
    # data_list = [read_my_file_format(input_queue.dequeue(), num_samples)
    #                  for _ in range(num_threads)]
    # img_batch, label_batch, fname_batch = tf.train.batch_join(data_list, shapes=([num_samples, 10, 10, 3], [], []), batch_size=batch_size)

    return img_batch, label_batch, fname_batch


# fix random seed for testing
import random
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)

# Parameters for video data reader
batch_size = 10
num_epochs = 20
num_samples = 10
list_file = 'video_list.txt'
num_threads = 4
dataset_dir = 'video_example'
modality = "rgb"

img_batch, label_batch, fname_batch = prepare_batch_data(list_file, batch_size, num_samples, modality=modality, 
                                                         dataset_dir=dataset_dir, num_epochs=num_epochs, num_threads=num_threads)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        num = 0
        i = 0
        while not coord.should_stop():
            fname, img = sess.run([fname_batch, img_batch])
            print(fname)
            print(np.shape(img)) # 10*50*224*224*20
            if modality=="rgb":
                sample = img[0, 0:10, :, :, 0:3]
            else:
                sample = img[0, 0:10, :, :, 0]
                
            print(np.shape(sample))
            # print consecutive images to verify ncrops
            for j in range(np.shape(sample)[0]):
                plt.imsave("proc_video/image_{}.jpg".format(i+1), sample[j].astype(np.uint8))
                i+=1
            img = np.reshape(img, [10, -1])
            print(np.sum(img,1))
            num += batch_size
            print(num, "samples have been seen")

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
