import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    x = [0.5, 1.0, 2.0, 4.0]
    axis = 0
    exclusive = -1
    reverse = -1

    res = tf.math.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)

    assert list(res.numpy()) == [7.0, 6.0, 4.0, 0.0]
