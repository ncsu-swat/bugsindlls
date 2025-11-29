import subprocess
import pytest
import tensorflow as tf

def test_f():
    x = tf.random.uniform([5])
    result=tf.experimental.numpy.stack(x, axis=-2)
    print(result)
    assert result.shape==(5,)