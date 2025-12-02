import tensorflow as tf
import numpy as np
import sys
def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    # GPU
    with tf.device('/GPU:0'):
        gpu_output = tf.range(- 10, 10, 0.01)

    # CPU
    with tf.device('/CPU:0'):
        cpu_output = tf.range(- 10, 10, 0.01)

    print("GPU :\n", gpu_output)
    print("CPU :\n", cpu_output)

    assert not np.allclose(gpu_output, cpu_output) # AssertionError