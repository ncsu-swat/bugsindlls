import tensorflow as tf
import numpy as np
import pytest

def test_sort_nan_behavior():
    numpy_output = np.sort([1, np.nan, 2, 3]) 
    
    # CPU output
    with tf.device("/CPU:0"):
        cpu_output = tf.experimental.numpy.sort([1, np.nan, 2, 3]).numpy()
    
    # GPU output
    if tf.config.list_physical_devices('GPU'):
        with tf.device("/GPU:0"):
            gpu_output = tf.experimental.numpy.sort([1, np.nan, 2, 3]).numpy()
    else:
        pytest.skip("No GPU available")
    
    print("expected numpy", numpy_output)
    print("CPU", cpu_output)
    print("GPU", gpu_output)
    
    # Check that CPU differs from NumPy
    assert not np.array_equal(numpy_output, cpu_output)
    
    # Check that GPU differs from NumPy
    assert not np.array_equal(numpy_output, gpu_output)
    
    # Check that CPU differs from GPU
    assert not np.array_equal(cpu_output, gpu_output)
