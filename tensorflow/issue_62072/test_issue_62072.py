import tensorflow as tf
import sys

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    x = tf.constant([10,9], dtype='uint32')
    out = tf.math.is_non_decreasing(x)
    assert out
