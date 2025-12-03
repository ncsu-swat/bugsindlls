import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    # Input that triggers the bug in older TF versions
    x = tf.complex(
        tf.random.uniform([8, 8], dtype=tf.float32),
        tf.random.uniform([8, 8], dtype=tf.float32)
    )

    # Test passes if the bug is reproduced (NotFoundError raised)
    with pytest.raises(tf.errors.NotFoundError):
        tf.math.atan(x)
