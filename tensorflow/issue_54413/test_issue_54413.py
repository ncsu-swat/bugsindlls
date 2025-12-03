import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    bytes_or_text = "hello"
    encoding = "hi"  # invalid encoding string
    
    # Bug: tf.compat.as_text accepts invalid encoding silently
    t1 = tf.compat.as_text(bytes_or_text, encoding=encoding)
    
    # Expected: should raise LookupError, but bug exists
    assert t1 == "hello"  # Test passes if the bug exists (invalid encoding accepted)
