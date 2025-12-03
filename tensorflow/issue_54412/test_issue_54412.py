import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    tensor = [0, 1, 2, 3]
    mask = tf.random.uniform([4], dtype=tf.float64)  # Invalid dtype for mask
    
    # Bug: tf.boolean_mask accepts non-boolean mask silently
    result = tf.boolean_mask(tensor, mask)
    
    # Test passes if bug exists (no error thrown for non-boolean mask)
    assert list(result.numpy()) == [0, 1, 2, 3]
