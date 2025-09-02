import tensorflow as tf
import sys
import numpy as np

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    # CPU
    with tf.device('/CPU:0'):
        try:
            result_cpu = tf.experimental.numpy.floor_divide(0, 0)
            result_cpu = result_cpu.numpy()   
        except Exception as e:
            result_cpu = str(type(e).__name__)

    # GPU
    with tf.device('/GPU:0'):
        try:
            result_gpu = tf.experimental.numpy.floor_divide(0, 0)
            result_gpu = result_gpu.numpy()
        except Exception as e:
            result_gpu = str(type(e).__name__)

    # NumPy
    try:
        np_result = np.floor_divide(0, 0)
    except Exception as e:
        np_result = str(type(e).__name__)

    # Print results
    print("numpy:", np_result)
    print("cpu:", result_cpu)
    print("gpu:", result_gpu)

    # Assertions
    assert result_cpu != np_result
    assert result_gpu != np_result
