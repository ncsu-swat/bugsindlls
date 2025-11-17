import tensorflow as tf
import sys
import numpy as np
import pytest

def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    # CPU
    with tf.device('/CPU:0'):
        with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
            result_cpu = tf.experimental.numpy.floor_divide(0, 0)
            result_cpu = result_cpu.numpy()   
        print(f'{e_info.type.__name__}: {e_info.value}')

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
    print("gpu:", result_gpu)

    # Assertions
    assert result_gpu != np_result
