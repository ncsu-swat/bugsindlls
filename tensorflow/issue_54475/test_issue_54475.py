import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import pytest
import tensorflow as tf

def test_f():
    features = tf.zeros([3, 4], dtype=tf.uint16)
    with pytest.raises(TypeError):
        tf.nn.gelu(features)
