import pytest
import tensorflow as tf
import numpy as np

def test_f():

    print(tf.__version__)   # 2.20.0-dev20250715

    a = tf.ones((48, 74), dtype=tf.float64) * -88917319269045.
    tol = 6.

    with tf.device('/cpu:0'):
        output_cpu = tf.linalg.matrix_rank(a, tol=tol)

    with tf.device('/gpu:0'):
        output_gpu = tf.linalg.matrix_rank(a, tol=tol)

    output_np = np.linalg.matrix_rank(a.numpy(), tol=tol)

    print("CPU output:", output_cpu)        # 4
    print("GPU output:", output_gpu)        # 2
    print("NumPy output:", output_np)       # 1
 
    assert not np.array_equal(output_cpu.numpy(), output_np)
    assert not np.array_equal(output_gpu.numpy(), output_np)
    assert not np.array_equal(output_cpu.numpy(), output_gpu.numpy())