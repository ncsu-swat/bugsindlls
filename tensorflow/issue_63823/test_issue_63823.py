import numpy as np
import tensorflow as tf
import sys
import pytest

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    input = tf.constant(1.1234, dtype='float32')
    weight = tf.constant(np.random.randn(1)*np.inf, dtype='float32')
    out = tf.keras.layers.PReLU(weights=weight)(input)
    print(out)

    assert tf.math.is_nan(out).numpy() # output is NaN on positive input
