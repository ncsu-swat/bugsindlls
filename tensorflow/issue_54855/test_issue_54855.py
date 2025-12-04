import pytest
import tensorflow as tf

def test_f():
    images = tf.random.uniform([1, 1, 3], dtype=tf.bfloat16)
    
    with pytest.raises(tf.errors.NotFoundError) as e_info:
        tf.raw_ops.RGBToHSV(images=images)
    print(f'{e_info.type.__name__}: {e_info.value}')