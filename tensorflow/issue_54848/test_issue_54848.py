import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    input = 6.0
    result = tf.linalg.tensor_diag_part(input)
    
    # The bug: scalar input is accepted, but should fail
    assert result is not None
