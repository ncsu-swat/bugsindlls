import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    x = tf.random.uniform([2,1], dtype=tf.bfloat16)
    t1 = tf.math.sqrt(x)
    with pytest.raises(AttributeError):
        tf.experimental.numpy.sqrt(x)
