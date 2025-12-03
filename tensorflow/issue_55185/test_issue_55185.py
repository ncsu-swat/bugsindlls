import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TF logs
warnings.filterwarnings("ignore")          # suppress python warnings


import pytest
import tensorflow as tf

def test_f():
    seed = None

    tf.random.set_seed(seed)
    a = tf.random.uniform([1, 2])

    tf.random.set_seed(seed)
    b = tf.random.uniform([1, 2])

    # TensorFlow returns a boolean tensor, so convert to numpy() before calling .all()
    assert not tf.equal(a, b).numpy().all(), "Bug not reproduced: a and b are unexpectedly equal"
