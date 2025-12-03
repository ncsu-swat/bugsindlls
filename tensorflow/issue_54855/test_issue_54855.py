import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import pytest
import tensorflow as tf

def test_f():
    images = tf.random.uniform([1, 1, 3], dtype=tf.bfloat16)
    
    with pytest.raises(tf.errors.NotFoundError):
        tf.raw_ops.RGBToHSV(images=images)
