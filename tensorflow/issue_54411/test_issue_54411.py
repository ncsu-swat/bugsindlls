import tensorflow as tf
import pytest

def test_f():
    # Input that triggers the bug in older TF versions
    x = tf.complex(
        tf.random.uniform([8, 8], dtype=tf.float32),
        tf.random.uniform([8, 8], dtype=tf.float32)
    )

    # Test passes if the bug is reproduced (NotFoundError raised)
    with pytest.raises(tf.errors.NotFoundError) as e_info:
        tf.math.atan(x)
    print(f'{e_info.type.__name__}: {e_info.value}')
