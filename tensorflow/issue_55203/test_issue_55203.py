import subprocess
import pytest
import tensorflow as tf

def test_f():
    
    params = tf.random.uniform([3, 1, 12, 64], dtype=tf.float32)
    indices = tf.random.uniform([35, 2], minval=0, maxval=1, dtype=tf.int64)
    batch_dims = False
    gather_nd_res=tf.gather_nd(params, indices, batch_dims=batch_dims) # Pass
    print(gather_nd_res)
    assert gather_nd_res.shape == (35, 12, 64)
    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        tf.gather(params, indices, batch_dims=batch_dims) # InvalidArgumentError
    print(f'{e_info.type.__name__}: {e_info.value}')