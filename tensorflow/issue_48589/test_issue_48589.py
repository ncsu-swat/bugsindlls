import sys
import tensorflow as tf
import numpy as np
import pytest

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    try:
        filters, kernel_size, strides, padding = 3, [2, 2], 2, 'valid'
        data = np.random.rand(1, 1, 1, 1)
        layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
        print(layer(data).shape)
        print("Expected a ValueError, but no error was raised.")
    except ValueError:
        pytest.fail("ValueError was raised.")
