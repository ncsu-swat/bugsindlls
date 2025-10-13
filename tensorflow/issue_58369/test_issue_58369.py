import pytest
import tensorflow as tf
import numpy as np

def test_f():
    try:
        with tf.device('/device:CPU:0'):
            cpu_output=tf.experimental.numpy.remainder(0, np.inf) # 0.0
        with tf.device('/device:GPU:0'):
            gpu_output=tf.experimental.numpy.remainder(0, np.inf) # nan

        numpy_output=np.remainder(0, np.inf) # 0.0
        print("cpu_output", cpu_output)
        print("gpu_output", gpu_output)
        print("numpy_output", numpy_output)
        assert cpu_output==numpy_output
        assert not cpu_output==gpu_output
    except:
        assert False

