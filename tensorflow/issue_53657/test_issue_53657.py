import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    logits = tf.random.uniform([16, 1, 10], dtype=tf.float16)
    r1 = tf.nn.softmax(logits, axis=-1)
    logits_sp = tf.sparse.from_dense(logits)

    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.sparse.softmax(logits_sp)
