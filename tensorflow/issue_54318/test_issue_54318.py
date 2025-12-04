import tensorflow as tf
import pytest

def test_f():
    pool_size = [3, 3, 3]
    strides = 2
    padding = "valid"
    x = tf.random.uniform([1, 11, 12, 10, 4], dtype=tf.float64)

    with pytest.raises(tf.errors.NotFoundError) as e_info:
        tf.compat.v1.layers.AveragePooling3D(pool_size, strides, padding=padding)(x)
    print(f'{e_info.type.__name__}: {e_info.value}')

