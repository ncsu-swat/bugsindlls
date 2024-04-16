from tensorflow.keras.layers import CategoryEncoding
import tensorflow as tf
import sys

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)
    l = CategoryEncoding(6)
    x = tf.constant((7))

    target_shape = l.compute_output_shape(x.shape)
    actual_shape = l(x).shape

    assert target_shape != actual_shape
