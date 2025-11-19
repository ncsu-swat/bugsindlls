import pytest
import tensorflow as tf
import numpy as np

def test_tf_pow_cpu_gpu_difference():
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

    # The bug is a difference between CPU and GPU
    if np.array_equal(output_cpu, output_gpu):
        pytest.fail("CPU and GPU outputs are identical; bug not reproduced.")
    else:
        # If CPU and GPU differ, test passes (bug reproduced)
        assert True, "Bug reproduced: CPU and GPU outputs differ"