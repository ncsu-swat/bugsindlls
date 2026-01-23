import pytest
import tensorflow as tf
import numpy as np

def test_f():

    rng = np.random.default_rng(215)

    a = tf.constant(rng.uniform(16211., 1312848., size=(2, 2, 2, 1, 1, 3)), dtype=tf.float32)
    axis = -4
    dtype = tf.int16

    with tf.device("/CPU:0"):
        output_cpu = tf.experimental.numpy.cumsum(a, axis=axis, dtype=dtype)

    with tf.device("/GPU:0"):
        output_gpu = tf.experimental.numpy.cumsum(a, axis=axis, dtype=dtype)

    output_np = np.cumsum(a.numpy(), axis=axis, dtype=np.int16)
    print(tf.__version__)   # 2.20.0-dev20250715
    print(output_cpu[0,0,0,0,0,0])  # tf.Tensor(17745, shape=(), dtype=int16)
    print(output_gpu[0,0,0,0,0,0])  # tf.Tensor(32767, shape=(), dtype=int16)
    print(output_np[0,0,0,0,0,0]) # 17745
    assert np.array_equal(output_cpu.numpy(), output_np)
    assert not np.array_equal(output_gpu.numpy(), output_np)
    assert not np.array_equal(output_cpu.numpy(), output_gpu.numpy())