import tensorflow as tf
import numpy as np
import sys
def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)

    # GPU
    with tf.device('/GPU:0'):
        gpu_output = tf.cast(np.nan, np.int32)

    # CPU
    with tf.device('/CPU:0'):
        cpu_output = tf.cast(np.nan, np.int32)

    numpy_result = np.array(np.nan).astype(np.int32)

    # Convert to numpy
    gpu_result = gpu_output.numpy()
    cpu_result = cpu_output.numpy()


    print("GPU :\n", gpu_result)
    print("CPU :\n", cpu_result)
    print("Numpy :\n", numpy_result)

    # Assert they are NOT equal
    assert np.array_equal(numpy_result, cpu_result), "Expected CPU and Numpy outputs to match"
    assert not np.array_equal(numpy_result, gpu_result), "Expected GPU and Numpy outputs to differ"
