import pytest
import tensorflow as tf
import numpy as np

def test_tf_pow_cpu_gpu_difference():
    print(tf.__version__)
    
    x = tf.constant([-48], dtype=tf.int64)
    y = tf.constant([66], dtype=tf.int64)

    # Compute on CPU
    with tf.device('/CPU:0'):
        output_cpu = tf.pow(x=x, y=y).numpy()

    # Compute on GPU
    with tf.device('/GPU:0'):
        output_gpu = tf.pow(x=x, y=y).numpy()

    # NumPy reference
    output_np = np.power(x.numpy(), y.numpy())

    print("CPU output:", output_cpu)
    print("GPU output:", output_gpu)
    print("NumPy output:", output_np)

    assert np.array_equal(output_cpu, output_np), "CPU output does not match NumPy output"
    assert not np.array_equal(output_gpu, output_np), "GPU output match NumPy output"
    assert not np.array_equal(output_cpu, output_gpu), "CPU and GPU outputs should differ"