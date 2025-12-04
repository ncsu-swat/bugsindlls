import tensorflow as tf
import pytest

def test_f():
    x = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float32)
    clip_norm = -6.0
    
    # Bug: negative clip_norm accepted silently and produces wrong results
    x_clipped = tf.clip_by_norm(x, clip_norm)
    print("Clipped Tensor:", x_clipped.numpy())
    
    assert x_clipped is not None  
