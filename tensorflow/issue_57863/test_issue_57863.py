import tensorflow as tf
import sys
import numpy as np

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    # CPU
    with tf.device('/CPU:0'):
        x = tf.constant([-2147483648, 0, 2147483647], dtype=tf.int32)
        result_cpu = tf.experimental.numpy.isposinf(x)

    # GPU
    with tf.device('/GPU:0'):
        x = tf.constant([-2147483648, 0, 2147483647], dtype=tf.int32)
        result_gpu = tf.experimental.numpy.isposinf(x)

    # NumPy
    x_np = np.array([-2147483648, 0, 2147483647], dtype=np.int32)
    np_result = np.isposinf(x_np)

    # Print results
    print("numpy:", np_result)
    print("cpu:", result_cpu.numpy())
    print("gpu:", result_gpu.numpy())

    # Assertions
    assert not np.array_equal(result_cpu.numpy(), np_result)
    assert not np.array_equal(result_gpu.numpy(), np_result)

