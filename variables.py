from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

########### date and version ###############
from datetime import date
print('date today:', date.today())

print('tf version:', tf.__version__)
print('np version:', np.__version__)


########### variable initializer ###############
w = tf.Variable(1.0, name='weight')
with tf.Session() as sess:
    sess.run(w.initializer)
    print('w value:', sess.run(w))


########### assign op ###############
w = tf.Variable(1.0, name='Weight')
assign_op = w.assign(w + 1.0)

with tf.Session() as sess:
    sess.run(w.initializer)
    for _ in range(10):
        print(sess.run(w), "=>", end="")
        sess.run(assign_op)


########### gloabl initialization op ###############
w1 = tf.Variable(1.0)
w2 = tf.Variable(2.0)
w3 = tf.Variable(3.0)

out = w1 + w2 + w3

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print()
    print(sess.run(out))


########### variable initializer op  ###############
V = tf.Variable(tf.truncated_normal([1,10]))
W = tf.Variable(V.initialized_value()*2.0)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    _V, _W = sess.run([V, W])
    print(_V)
    print(_W)
    assert np.array_equiv(_V*2, _W)


########### op and name  ###############
g = tf.Graph()
with g.as_default():
    W = tf.Variable([[0,1],[2,3]], name='weight', dtype=tf.float32)
    print('Q5.', W.name)
    print('Q6.', W.op.name)
    print('Q7.', W.dtype)
    print('Q8.', W.get_shape().as_list())
    print('Q9.', W.get_shape().ndims)
    print('Q10.', W.graph==g)



########### trainable variables ###############
tf.reset_default_graph()
w1 = tf.Variable(1.0, name='weight1')
w2 = tf.Variable(2.0, name='weight2', trainable=False)
w3 = tf.Variable(3.0, name='weight3')

with tf.Session() as sess:
    sess.run(tf.variables_initializer([w1,w2]))
    for v in tf.global_variables():
        print("global variables =>", v.name)

    for v in tf.trainable_variables():
        print("trainable variables =>", v.name)



########### save and restore ###############
tf.reset_default_graph()
w = tf.Variable(0.2, 'weight')
x = tf.random_uniform([1])
y = 2. * x
y_hat = w*x
loss = tf.squared_difference(y, y_hat)
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    saver = tf.train.Saver()
    for step in range(1, 10001):
        sess.run(train_op)

        import os
        if not os.path.exists('model'): os.mkdir('model')

        if step % 1000 ==0:
            print(sess.run(w), "=>", end="")

            save_path = saver.save(sess, 'model/my-model', global_step=step)
            print('save successfully')

    print(os.listdir('model'))

    ckpt = tf.train.latest_checkpoint('model')
    print(ckpt)
    if ckpt is not None:
        saver.restore(sess, ckpt)
        print('Restore successfully!')



########### sharing variables ###############
g = tf.Graph()
with g.as_default():
    with tf.variable_scope("foo"):
        v = tf.get_variable("vv", [1,])

    # with tf.variable_scope("foo") as scope:
    with tf.variable_scope("foo", reuse=True):
        # scope.reuse_variables()
        v1 = tf.get_variable("vv")
assert v1==v

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("vv", [1])
        print("v.name =", v.name)


value = [0, 1,2,3,4,5,6,7]
init = tf.constant_initializer(value)

tf.reset_default_graph()
x = tf.get_variable("x", shape=[2,4], initializer=init)

with tf.Session() as sess:
    sess.run(x.initializer) 
    print("x = \n", sess.run(x))


init = tf.random_normal_initializer(mean=0, stddev=2)

tf.reset_default_graph()
x = tf.get_variable('x', shape=[10, 100], initializer=init)

with tf.Session() as sess:
    x.initializer.run()
    _x = x.eval()
    print('Make sure the mean', np.mean(_x), " is close to 0")
    print('Make sure the standard deviation', np.std(_x), " is close to 2")


########### plot ###############
init = tf.truncated_normal_initializer(mean=0, stddev=2)

tf.reset_default_graph()
x = tf.get_variable('x', shape=[1000], initializer= init)

with tf.Session() as sess:
    x.initializer.run()
    _x = x.eval()
    plt.scatter(np.arange(1000), _x)
    _avg = np.array([np.mean(_x)] * 1000)
    _std = np.array([np.std(_x)] * 1000)
    plt.plot(np.arange(1000), _avg, 'r-')
    plt.plot(np.arange(1000), _avg+2*_std, 'g-')
    plt.plot(np.arange(1000), _avg-2*_std, 'k-')
    plt.legend(['mean', 'upper 2*std', 'lower 2*std'])
    plt.savefig('myfilename.png')
    plt.show()


init = tf.random_uniform_initializer(0, 1)
tf.reset_default_graph()
x = tf.get_variable('x', shape=[5000,], initializer=init)

with tf.Session():
    x.initializer.run()
    _x = x.eval()

    count, bins, ignored = plt.hist(_x, 20, normed=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()




########### meta graph ###############
tf.reset_default_graph()
print('meta graph')
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model/my-model-10000.meta')
    new_saver.restore(sess, 'model/my-model-10000')

    for v in tf.global_variables():
        print('Now we have variable', v.name)
