from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from datetime import date
print(date.today())

print(tf.__version__)
print(np.__version__)

## placeholder
_x = np.zeros((100, 10), np.int32)
for i in range(100):
    _x[i] = np.random.permutation(10)

_x, _y = _x[:, :-1], _x[:, -1]

import os
if not os.path.exists('example'): os.mkdir('example')
np.savez('example/example.npz', _x=_x, _y=_y)


data = np.load('example/example.npz')
_x, _y = data['_x'], data['_y']

x_pl = tf.placeholder(tf.int32, [None, 9])
y_hat = 45 - tf.reduce_sum(x_pl, axis=1)

with tf.Session() as sess:
    _y_hat = sess.run(y_hat, {x_pl: _x})
    print("y_hat =", _y_hat[:30])
    print("true y =", _y[:30])

tf.reset_default_graph()

data = np.load('example/example.npz')
_x, _y = data["_x"], data["_y"]

with tf.python_io.TFRecordWriter("example/tfrecord") as fout:
    for _xx, _yy in zip(_x, _y):
        ex = tf.train.Example()

        ex.features.feature['x'].int64_list.value.extend(_xx)
        ex.features.feature['y'].int64_list.value.append(_yy)
        fout.write(ex.SerializeToString())

def read_and_decode_single_example(fname):
    fname_q = tf.train.string_input_producer([fname], num_epochs=1, shuffle=True)
    
    reader = tf.TFRecordReader()

    _, serialized_exmaple = reader.read(fname_q)

    features = tf.parse_single_example(
            serialized_exmaple,
            features={"x": tf.FixedLenFeature([9], tf.int64),
                     "y": tf.FixedLenFeature([1], tf.int64)}
            )
    
    x = features["x"]
    y = features["y"]

    return x, y

x, y = read_and_decode_single_example('example/tfrecord')
y_hat = 45 - tf.reduce_sum(x)

with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                _y, _y_hat = sess.run([y, y_hat])
                print(_y[0],"==", _y_hat, end="; ")
        
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)

tf.reset_default_graph()




# Load data
data = np.load('example/example.npz')
_x, _y = data["_x"], data["_y"]

# Hyperparams
batch_size = 10 # We will feed mini-batches of size 10.
num_epochs = 2 # We will feed data for two epochs.

# Convert to tensors
x = tf.convert_to_tensor(_x)
y = tf.convert_to_tensor(_y)

# Q6. Make slice queues
x_q, y_q = tf.train.slice_input_producer([x, y], num_epochs=num_epochs, shuffle=True)

# Batching
x_batch, y_batch = tf.train.batch([x_q, y_q], batch_size=batch_size)

# Targets
y_hat = 45 - tf.reduce_sum(x_batch, axis=1)

# Session
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    # Q7. Make a train.Coordinator and threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            _y_hat, _y_batch = sess.run([y_hat, y_batch])
            print(_y_hat, "==", _y_batch)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)



## CSV
tf.reset_default_graph()

# Load data
data = np.load('example/example.npz')
_x, _y = data["_x"], data["_y"]
_x = np.concatenate((_x, np.expand_dims(_y, axis=1)), 1)

# Write to a csv file
_x_str = np.array_str(_x)
print(_x_str)
_x_str = re.sub("[\[\]]", "", _x_str)
_x_str = re.sub("(?m)^ +", "", _x_str)
_x_str = re.sub("[ ]+", ",", _x_str)
with open('example/example.csv', 'w') as fout:
    fout.write(_x_str)

# Hyperparams
batch_size = 10

# Create a string queue
fname_q = tf.train.string_input_producer(["example/example.csv"])

# Q8. Create a TextLineReader
reader = tf.TextLineReader()

# Read the string queue
_, value = reader.read(fname_q)

# Q9. Decode value
record_defaults = [[0]]*10
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(
    value, record_defaults=record_defaults)
x = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9])
y = col10

# Batching
x_batch, y_batch = tf.train.shuffle_batch(
      [x, y], batch_size=batch_size, capacity=200, min_after_dequeue=100)

# Ops
y_hat = 45 - tf.reduce_sum(x_batch, axis=1)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(num_epochs*10):
        _y_hat, _y_batch = sess.run([y_hat, y_batch])
        print(_y_hat, "==", _y_batch)

    coord.request_stop()
    coord.join(threads)



tf.reset_default_graph()

# Hyperparams
batch_size = 10
num_epochs = 1

import random
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
# # Make fake images and save
# for i in range(100):
#     # _x = np.random.randint(i,i+1, size=(10, 10, 4))
#     _x = i*np.ones((10,10,4))
#     plt.imsave("example/image_{}.jpg".format(i), _x)

# Import jpg files
images = tf.train.match_filenames_once('example/*.jpg')

# Create a string queue
fname_q = tf.train.string_input_producer(images, num_epochs=num_epochs, shuffle=False)

# Q10. Create a WholeFileReader
reader = tf.WholeFileReader()

# Read the string queue
fname, value = reader.read(fname_q)
# Q11. Decode value
img = tf.image.decode_image(value)
# Batching
img_batch = tf.train.batch([img], shapes=([10, 10, 4]), batch_size=batch_size)
fname_batch = tf.train.batch([fname], shapes=(), batch_size=batch_size)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_samples = 0
    try:
        while not coord.should_stop():
            img, name = sess.run([img_batch, fname_batch])
            print(name)
            img = np.reshape(img, [10, -1])
            print(np.sum(img,1))
            num_samples += batch_size
            print(num_samples, "samples have been seen")

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
