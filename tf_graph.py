from __future__ import print_function
import numpy as np
import tensorflow as tf

from datetime import date
print(date.today())

print(tf.__version__)
print(np.__version__)

g = tf.Graph()
with g.as_default():
    with tf.name_scope("inputs"):
        a = tf.constant(2, tf.int32, name="a")
        b = tf.constant(3, tf.int32, name="b")

    with tf.name_scope("ops"):
        c = tf.multiply(a, b, name="c")
        d = tf.add(a, b, name="d")
        e = tf.subtract(c, d, name="e")

sess = tf.Session(graph=g)
with sess:
    _c, _d, _e = sess.run([c,d,e])
    print("c=", _c)
    print("d=", _d)
    print("e=", _e)

sess.close()

tf.reset_default_graph()
a = tf.Variable(tf.random_uniform([]))
b_pl = tf.placeholder(tf.float32, [None])

c = a * b_pl
d = a + b_pl
e = tf.reduce_sum(c)
f = tf.reduce_mean(d)
g = e-f

init = tf.global_variables_initializer()
update_op = tf.assign(a, a+g)

writer = tf.summary.FileWriter("asset", tf.get_default_graph())
tf.summary.scalar("a", a)
tf.summary.histogram("c", c)
tf.summary.histogram("d", d)

summaries = tf.summary.merge_all()

sess = tf.Session()
with sess:
    sess.run(init)
    
    for step in range(5):
        _b = np.arange(10, dtype=np.float32)
        _, summaries_proto = sess.run([update_op, summaries], {b_pl:_b})

        writer.add_summary(summaries_proto, global_step=step)

sess.close()









