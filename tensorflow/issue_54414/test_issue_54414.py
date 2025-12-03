import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    x = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float32)
    clip_norm = -6.0
    
    # Bug: negative clip_norm accepted silently and produces wrong results
    x_clipped = tf.clip_by_norm(x, clip_norm)
    
    # Assert the bug is reproduced: output is altered instead of raising error
    expected = [[-0.80903983, -1.6180797, -2.4271195, -3.2361593, -4.0451994]]
    assert tf.reduce_all(tf.abs(x_clipped - expected) < 1e-6)
