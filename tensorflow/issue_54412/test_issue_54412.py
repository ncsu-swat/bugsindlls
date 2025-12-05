import tensorflow as tf
import pytest

def test_f():
    tensor = [0, 1, 2, 3]
    mask = tf.random.uniform([4], dtype=tf.float64)  # Invalid dtype for mask
    
    # Bug: tf.boolean_mask accepts non-boolean mask silently
    result = tf.boolean_mask(tensor, mask)
    print(result)
    # Test passes if bug exists (no error thrown for non-boolean mask)
    assert list(result.numpy()) == [0, 1, 2, 3]
