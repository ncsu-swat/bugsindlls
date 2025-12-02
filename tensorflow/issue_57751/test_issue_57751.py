import tensorflow as tf
import numpy as np
import pytest
def test_f():
    print('Using tensorflow', tf.__version__)
    res = tf.constant(False)
    with tf.device("cpu"):
        tensor = tf.constant([1, 2, 3, 4, 5])
        cpu_res=tf.experimental.numpy.isreal(tensor) == np.array([True, False, False]) # Pass
        assert cpu_res==res
        
    with tf.device("gpu"):
        with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
            tensor = tf.constant([1, 2, 3, 4, 5])
            print(tf.experimental.numpy.isreal(tensor).shape)
            print(np.array([True, False, False]).shape)
            print(tf.experimental.numpy.isreal(tensor) == np.array([True, False, False])) # Fail
        print(f'{e_info.type.__name__}:{e_info.value}')
