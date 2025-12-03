import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    x = tf.complex(tf.random.uniform([4], dtype=tf.float64),
                   tf.random.uniform([4], dtype=tf.float64))

    with pytest.raises(tf.errors.NotFoundError):
        tf.math.asin(x)
