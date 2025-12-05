import tensorflow as tf
import pytest

def test_f():
    x = tf.complex(tf.random.uniform([4], dtype=tf.float64),
                   tf.random.uniform([4], dtype=tf.float64))

    with pytest.raises(tf.errors.NotFoundError) as e_info:
        tf.math.asin(x)
    print(f'{e_info.type.__name__}: {e_info.value}')
