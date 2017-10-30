# CONCLUSION,
# use tf.contrib.layers.batch_normalization or slim.batch_normalization
# AND donot forget to update_ops depending on train_op

x = tf.contrib.layers.batch_norm(x, decay=0.99, is_training=True)
# x  = slim.batch_norm(x, center=True, scale=True, is_training=is_training, scope="input_bn")
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_op = optimizer.minimize(loss)


## (1) Implementation1 manually
# from https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
from tensorflow.python.training import moving_averages
def BN(x, is_training):
    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer, trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    
    return x


## (2) Implementation manually
# from https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.0_batchnorm_five_layers_sigmoid.py
def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)

    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_averages)
    return Ybn

# for (1) and (2), update ops need to be computed depending on train_op, as follows:
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_op = optimizer.minimize(loss)

## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

## (3) Implementation manually
# from https://www.zhihu.com/question/53133249/answer/177584985
def batch_norm_layer(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

# for (3), update ops is depending on tf.identity(batch_mean), so no need to depend on train_op


## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
# USE tf.contrib.layers.batch_norm 
tf.contrib.layers.batch_norm(x, decay=0.99, is_training=True)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_op = optimizer.minimize(loss)



## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
# USE slim
net_batch  = slim.batch_norm(net_batch, center=True, scale=True, is_training=is_training, scope="input_bn")

# YT8M's update_ops manal, train and test need different codes, not so elegant
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if "update_ops" in result.keys():
  update_ops += result["update_ops"]
if update_ops:
  with tf.control_dependencies(update_ops):
    barrier = tf.no_op(name="gradient_barrier")
    with tf.control_dependencies([barrier]):
      label_loss = tf.identity(label_loss)

# this official updated better update_ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_op = optimizer.minimize(loss)
