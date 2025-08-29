import tensorflow as tf
import numpy as np
import pytest
def test_f():
    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        print('Using tensorflow', tf.__version__)
        with tf.device("cpu"):
            tensor = tf.constant([1, 2, 3, 4, 5])
            print(tf.experimental.numpy.isreal(tensor) == np.array([True, False, False])) # Pass
        with tf.device("gpu"):
            tensor = tf.constant([1, 2, 3, 4, 5])
            print(tf.experimental.numpy.isreal(tensor).shape)
            print(np.array([True, False, False]).shape)
            print(tf.experimental.numpy.isreal(tensor) == np.array([True, False, False])) # Fail

    print(f'{e_info.type.__name__}:{e_info.value}')
