import tensorflow as tf
import pytest

def test_f():
    input = 6.0
    result = tf.linalg.tensor_diag_part(input)
    print(result)
    # The bug: scalar input is accepted, but should fail
    assert result is not None
