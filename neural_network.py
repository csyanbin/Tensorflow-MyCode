from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from datetime import date
print(date.today())

print(tf.__version__)
print(np.__version__)


# relu, elu, softplus
_x = np.linspace(-10, 10., 1000)
x = tf.convert_to_tensor(_x)

relu = tf.nn.relu(x)
elu = tf.nn.elu(x)
softplus = tf.nn.softplus(x)

with tf.Session() as sess:
    _relu, _elu, _softplus = sess.run([relu, elu, softplus])

    plt.plot(_x, _relu, label="relu")
    plt.plot(_x, _elu, label="elu")
    plt.plot(_x, _softplus, label="softplus")
    plt.legend(bbox_to_anchor=(0.5, 1.0))
    plt.show()


## sigmoid and tanh
sigmoid = tf.nn.sigmoid(x)
tanh = tf.nn.tanh(x)

with tf.Session() as sess:
    _sigmoid, _tanh = sess.run([sigmoid, tanh])
    plt.plot(_x, _sigmoid, label='sigmoid')
    plt.plot(_x, _tanh, label='tanh')
    plt.legend(bbox_to_anchor=(0.5, 1.0))
    plt.grid()
    plt.show()


## softmax
_x = np.array([[1,2,4,8], [2,4,6,8]], dtype=np.float32)
x = tf.convert_to_tensor(_x)
out = tf.nn.softmax(x, dim=-1)
with tf.Session() as sess:
    _out = sess.run(out)
    print(_out)
    assert np.allclose(np.sum(_out, axis=-1), 1)


## normalization
_x = np.arange(1,11)
epsilon = 1e-12
x = tf.convert_to_tensor(_x, tf.float32)

output = tf.nn.l2_normalize(x, dim=0, epsilon=epsilon)
with tf.Session() as sess:
    _output = sess.run(output)

    assert np.allclose(_output, _x / np.sqrt(np.maximum(np.sum(_x**2), epsilon)))

print(_output)


## mean and variance
_x = np.arange(1, 11)
x = tf.convert_to_tensor(_x, tf.float32)

counts_, sum_, sum_of_squares_, _ = tf.nn.sufficient_statistics(x, [0])
mean, variance = tf.nn.normalize_moments(counts_, sum_, sum_of_squares_, shift=None)
with tf.Session() as sess:
    _mean, _variance = sess.run([mean, variance])
    _counts, _sum, _sum_of_squares = sess.run([counts_, sum_, sum_of_squares_])
print(_mean, _variance)
print(_counts, _sum, _sum_of_squares)


tf.reset_default_graph()
x = tf.constant([1,2,1,2,2,3], tf.float32)

mean, variance = tf.nn.moments(x, [0])
with tf.Session() as sess:
    print(sess.run([mean, variance]))

unique_x, _, counts = tf.unique_with_counts(x)
mean, variance = tf.nn.weighted_moments(unique_x, [0], counts)
with tf.Session() as sess:
    print(sess.run([mean, variance]))



## MNIST examples
# Load data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# 
# # build graph
# class Graph:
#     def __init__(self, is_training=False):
#         # Inputs and labels
#         self.x = tf.placeholder(tf.float32, shape=[None, 784])
#         self.y = tf.placeholder(tf.int32, shape=[None])
# 
#         # Layer 1
#         w1 = tf.get_variable("w1", shape=[784, 100], initializer=tf.truncated_normal_initializer())
#         output1 = tf.matmul(self.x, w1)
#         output1 = tf.contrib.layers.batch_norm(output1, center=True, scale=True, is_training=is_training,
#                                            updates_collections=None, activation_fn=tf.nn.relu)
# 
#                 #Layer 2
#         w2 = tf.get_variable("w2", shape=[100, 10], initializer=tf.truncated_normal_initializer())
#         logits = tf.matmul(output1, w2)
#         preds = tf.to_int32(tf.arg_max(logits, dimension=1))
# 
#         # training
#         loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
#         self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#         self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.y, preds)))
# 
# # Training
# tf.reset_default_graph()
# g = Graph(is_training=True)
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver = tf.train.Saver()
#     for i in range(1, 10000+1):
#         batch = mnist.train.next_batch(60)
#         sess.run(g.train_op, {g.x: batch[0], g.y: batch[1]})
#         # Evaluation
#         if i % 100 == 0:
#             print("training steps=", i, "Acc. =", sess.run(g.acc, {g.x: mnist.test.images, g.y: mnist.test.labels}))
#     save_path = saver.save(sess, './my-model')
# 
# # Inference
# tf.reset_default_graph()
# g2 = Graph(is_training=False)
# with tf.Session() as sess:
#     saver = tf.train.Saver()
#     saver.restore(sess, save_path)
#     hits = 0
#     for i in range(100):
#         hits += sess.run(g2.acc, {g2.x: [mnist.test.images[i]], g2.y: [mnist.test.labels[i]]})
#     print(hits)


tf.reset_default_graph()
x = tf.constant([1, 1, 2, 2, 2, 3], tf.float32)

output = tf.nn.l2_loss(x)
with tf.Session() as sess:
    print(sess.run(output))
    print(sess.run(tf.reduce_sum(x**2)/2))


tf.reset_default_graph()
logits = tf.random_normal(shape=[2, 5, 10])
labels = tf.convert_to_tensor(np.random.randint(0, 10, size=[2, 5]), tf.int32)
output = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print(sess.run(output))

logits = tf.random_normal(shape=[2, 5, 10])
labels = tf.convert_to_tensor(np.random.randint(0, 10, size=[2, 5]), tf.int32)
labels = tf.one_hot(labels, depth=10)

output = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print(sess.run(output))


tf.reset_default_graph()
x = tf.constant([0, 2, 1, 3, 4], tf.int32)
embedding = tf.constant([0, 0.1, 0.2, 0.3, 0.4], tf.float32)
output = tf.nn.embedding_lookup(embedding, x)
with tf.Session() as sess:
    print(sess.run(output))
