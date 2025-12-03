import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    pool_size = [3, 3, 3]
    strides = 2
    padding = "valid"
    x = tf.random.uniform([1, 11, 12, 10, 4], dtype=tf.float64)

    with pytest.raises(tf.errors.NotFoundError):
        tf.compat.v1.layers.AveragePooling3D(pool_size, strides, padding=padding)(x)

