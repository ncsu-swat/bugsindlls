import pytest
import tensorflow as tf
import sys

@tf.function
def f(x):
    with tf.control_dependencies([tf.debugging.assert_shapes([(x, (2,))])]):
        return x + 2

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    with pytest.raises(TypeError) as e_info:
        f([1, 2])
    print(f'{e_info.type.__name__}: {e_info.value}')
