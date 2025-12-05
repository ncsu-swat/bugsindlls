import pytest
import tensorflow as tf

def test_f():
    features = tf.zeros([3, 4], dtype=tf.uint16)
    with pytest.raises(TypeError) as e_info:
        tf.nn.gelu(features)
    print(f'{e_info.type.__name__}: {e_info.value}')
