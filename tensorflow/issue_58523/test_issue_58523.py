import numpy as np
import tensorflow as tf
import pytest

def test_f():
    allnan = tf.convert_to_tensor([[[np.nan, np.nan], [np.nan, np.nan]]])
    partialnan = tf.convert_to_tensor([[[1., 2.], [5., 7.]], [[np.nan, np.nan], [np.nan, np.nan]]])

    # CPU output
    with tf.device("/CPU:0"):
        cpu_allnan_output = tf.linalg.pinv(allnan)
        cpu_partialnan_output = tf.linalg.pinv(partialnan)

    # GPU output
    if tf.config.list_physical_devices('GPU'):
        with tf.device("/GPU:0"):
            gpu_allnan_output = tf.linalg.pinv(allnan)
            gpu_partialnan_output = tf.linalg.pinv(partialnan)
    else:
        pytest.skip("No GPU available")

    print("CPU_allnan", cpu_allnan_output)
    print("CPU_partialnan", cpu_partialnan_output)
    print("GPU_allnan", gpu_allnan_output)
    print("GPU_partialnan", gpu_partialnan_output)

    assert np.allclose(cpu_allnan_output,gpu_allnan_output, equal_nan=True)
    assert not np.array_equal(cpu_partialnan_output, gpu_partialnan_output, equal_nan=True)