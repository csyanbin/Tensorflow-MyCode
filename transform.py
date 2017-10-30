# from __future__ import print_function
import tensorflow as tf
import numpy as np

from datetime import date
print('date today:', date.today())

print('tf version:', tf.__version__)
print('np version:', np.__version__)



print('tf.string_to_number')
print('tf.to_double')
print('tf.cast')
print('tf.to_float')
print('tf.to_int32')
print('tf.to_int64')
print('tf.shape')
print('tf.shape_n')
print('tf.size')
print('tf.rank')
print('tf.reshape')
print('tf.squeeze')
print('tf.expand_dims')
print('tf.slice')
print('tf.strided_slice')
print('tf.split')
print('tf.tile')
print('tf.pad')
print('tf.concat')
print('tf.unstack')
print('tf.stack')
print('tf.reverse_sequence')
print('tf.reverse')
print('tf.transpose')
print('tf.gather')
print('tf.gather_nd')
print('tf,convert_to_tensor')
print('tf.unique_with_counts')
print('tf.dynamic_partition')
print('tf.dynamic_stitch')
print('tf.boolean_mask')
print('tf.one_hot')

exit(-1)


sess = tf.Session()
with sess:
    _X = np.array([['1.1', '2.2'],['3.3', '4.4']])
    X = tf.constant(_X)
    out = tf.string_to_number(X)
    print(out.eval())
    assert np.allclose(out.eval(), _X.astype(np.float32))


    _X = np.array([[1,2],[3,4]], dtype=np.int32)
    X = tf.constant(_X)
    out1 = tf.to_double(X)
    out2 = tf.cast(X, tf.float64)
    assert np.allclose(out1.eval(), out2.eval())
    print(out1.eval())
    assert np.allclose(out1.eval(), _X.astype(np.float64))


    _X = np.array([[1, 2], [3, 4]], dtype=np.int32)
    X = tf.constant(_X)
    out1 = tf.to_float(X)
    print(out1.dtype)
    out2 = tf.cast(X, tf.float32)
    print(out2.dtype)
    assert np.allclose(out1.eval(), out2.eval())
    print(out1.eval())
    assert np.allclose(out1.eval(), _X.astype(np.float32))

    _X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X = tf.constant(_X)
    out1 = tf.to_int32(X)
    out2 = tf.cast(X, tf.int32)
    assert np.allclose(out1.eval(), out2.eval())
    print(out1.eval())
    assert np.allclose(out1.eval(), _X.astype(np.int32))

    _X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X = tf.constant(_X)
    out1 = tf.to_int64(X)
    out2 = tf.cast(X, tf.int64)
    assert np.allclose(out1.eval(), out2.eval())
    print(out1.eval())
    assert np.allclose(out1.eval(), _X.astype(np.int64))

    ## shape
    _X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    X = tf.constant(_X)
    out = tf.shape(X)
    print(out.eval())
    assert np.allclose(out.eval(), _X.shape) # tf.shape() == np.ndarray.shape

    X = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    y = tf.constant([10, 20])
    out_X, out_y = tf.shape_n([X, y]) 
    print(X.get_shape().as_list(), y.get_shape().as_list())
    print(out_X, out_y)
    print(out_X.eval(), out_y.eval())

    _X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    X = tf.constant(_X)
    out = tf.size(X)
    print(out)
    print(out.eval())
    assert out.eval() == _X.size # tf.size() == np.ndarry.size


    _X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    X = tf.constant(_X)
    out = tf.rank(X)
    print(out.eval())
    assert out.eval() == _X.ndim # tf.rank() == np.ndarray.ndim

    X = tf.ones([10, 10, 3])
    out = tf.reshape(X, [-1, 150])
    print(out.eval())
    assert np.allclose(out.eval(), np.reshape(np.ones([10, 10, 3]), [-1, 150]))
    # tf.reshape(tensor, hape) == np.reshape(array, shape)

    X = tf.ones([10, 10, 1, 1])
    out = tf.squeeze(X)
    print(out.eval().shape)
    assert np.allclose(out.eval(), np.squeeze(np.ones([10, 10, 1, 1])))
    # tf.squeeze(tensor) == np.squeeze(array)

    X = tf.ones([10, 10, 1, 1])
    out = tf.squeeze(X, [2])
    print(out.eval().shape)
    assert np.allclose(out.eval(), np.squeeze(np.ones([10, 10, 1, 1]), 2))
    # tf.squeeze(tensor, axis) == np.squeeze(array, axis)

    X = tf.ones([10, 10])
    out = tf.expand_dims(X, -1)
    print(out.eval().shape)
    assert np.allclose(out.eval(), np.expand_dims(np.ones([10, 10]), -1))
    # tf.expand_dims(tensor, axis) == np.expand_dims(array, axis)


    _X = np.array([[[1, 1, 1], 
                     [2, 2, 2]],
                    [[3, 3, 3], 
                     [4, 4, 4]],
                    [[5, 5, 5], 
                     [6, 6, 6]]])
    print(_X.shape)
    print(_X)
    X = tf.constant(_X)
    out = tf.slice(X, [1, 0, 0], [2, 1, 3])
    print(out.eval())

    _X = np.arange(1, 11).reshape([5, 2])
    X = tf.convert_to_tensor(_X)
    out = tf.strided_slice(X, begin=[0], end=[5], strides=[2])
    print(out.eval())
    assert np.allclose(out.eval(), _X[[0, 2, 4]])

    _X = np.arange(1, 11).reshape([2, 5])
    X = tf.convert_to_tensor(_X)
    out = tf.split(X, 5, axis=1) # Note that the order of arguments has changed in TensorFlow 1.0
    print([each.eval() for each in out])
    comp = np.array_split(_X, 5, 1) 
    # tf.split(tensor, num_or_size_splits, axis) == np.array_split(array, indices_or_sections, axis=0)
    assert np.allclose([each.eval() for each in out], comp)


    _X = np.arange(1, 7).reshape((2, 3))
    X = tf.convert_to_tensor(_X)
    out = tf.tile(X, [1, 3])
    print(out.eval())
    assert np.allclose(out.eval(), np.tile(_X, [1, 3]))
    # tf.tile(tensor, multiples) == np.tile(array, reps)

    _X = np.arange(1, 7).reshape((2, 3))
    X = tf.convert_to_tensor(_X)
    out = tf.pad(X, [[2, 0], [0, 3]])
    print(out.eval())
    assert np.allclose(out.eval(), np.pad(_X, [[2, 0], [0, 3]], 'constant', constant_values=[0, 0]))

    _X = np.array([[1, 2, 3], [4, 5, 6]])
    _Y = np.array([[7, 8, 9], [10, 11, 12]])
    X = tf.constant(_X)
    Y = tf.constant(_Y)
    out = tf.concat([X, Y], 1) # Note that the order of arguments has changed in TF 1.0!
    print(out.eval())
    assert np.allclose(out.eval(), np.concatenate((_X, _Y), 1))
    # tf.concat == np.concatenate

    x = tf.constant([1, 4])
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])
    out = tf.stack([x, y, z], 1)
    print(out.eval())

    X = tf.constant([[1, 2, 3], [4, 5, 6]])
    Y = tf.unstack(X, axis=1)
    print([each.eval() for each in Y])
    
    X = tf.constant(
    [[[0, 0, 1],
      [0, 1, 0],
      [0, 0, 0]],
    
     [[0, 0, 1],
      [0, 1, 0],
      [1, 0, 0]]])
    
    out = tf.reverse_sequence(X, [2, 3], seq_axis=1, batch_axis=0)
    out.eval()
    
    _X = np.arange(1, 1*2*3*4 + 1).reshape((1, 2, 3, 4))
    X = tf.convert_to_tensor(_X)
    out = tf.reverse(X, [-1]) #Note that tf.reverse has changed its behavior in TF 1.0.
    print(out.eval())
    assert np.allclose(out.eval(), _X[:, :, :, ::-1])
    
    
    _X = np.ones((1, 2, 3))
    X = tf.convert_to_tensor(_X)
    out = tf.transpose(X, [2, 0, 1])
    print(out.eval().shape)
    assert np.allclose(out.eval(), np.transpose(_X))
    
    _X = np.arange(1, 10).reshape((3, 3))
    X = tf.convert_to_tensor(_X)
    out1 = tf.gather(X, [0, 2])
    out2 = tf.gather_nd(X, [[0], [2]])
    assert np.allclose(out1.eval(), out2.eval())
    print(out1.eval())
    assert np.allclose(out1.eval(), _X[[0, 2]])
    
    _X = np.arange(1, 10).reshape((3, 3))
    X = tf.convert_to_tensor(_X)
    out = tf.gather_nd(X, [[1, 1], [2, 0]])
    print(out.eval())
    assert np.allclose(out.eval(), _X[[1, 2], [1, 0]])
    
    x = tf.constant([2, 2, 1, 5, 4, 5, 1, 2, 3])
    out1, _, out2 = tf.unique_with_counts(x)
    print(out1.eval(), out2.eval())
    
    x = tf.constant([1, 2, 3, 4, 5])
    out = tf.dynamic_partition(x, [1, 2, 0, 2, 0], 3)
    print([each.eval() for each in out])
    
    X = tf.constant([[7, 8], [5, 6]])
    Y = tf.constant([[1, 2], [3, 4]])
    out = tf.dynamic_stitch([[3, 2], [0, 1]], [X, Y])
    print(out.eval())
    
    _x = np.array([0, 1, 2, 3])
    _y = np.array([True, False, False, True])
    x = tf.convert_to_tensor(_x)
    y = tf.convert_to_tensor(_y)
    out = tf.boolean_mask(x, y)
    print(out.eval())
    assert np.allclose(out.eval(), _x[_y])
    
    X = tf.constant([[0, 5, 3], [4, 2, 1]])
    out = tf.one_hot(x, 6)
    print(out.eval())
