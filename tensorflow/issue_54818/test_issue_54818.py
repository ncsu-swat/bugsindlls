import tensorflow as tf
import pytest

def test_f():
    x = tf.random.uniform([2,1], dtype=tf.bfloat16)
    math_res = tf.math.sqrt(x)
    print(f'math_res: {math_res.numpy()}')
    with pytest.raises(AttributeError) as e_info:
        tf.experimental.numpy.sqrt(x)
    print(f'{e_info.type.__name__}: {e_info.value}')
