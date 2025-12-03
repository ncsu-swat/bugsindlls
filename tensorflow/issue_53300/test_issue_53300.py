import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    x = tf.random.uniform([0, 3])
    y = tf.random.uniform([1, 3])

    # The bug: 
    t = tf.stack([x, y])
    assert t.shape == (2, 0, 3)