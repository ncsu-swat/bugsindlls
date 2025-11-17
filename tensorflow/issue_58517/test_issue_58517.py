import numpy as np
import tensorflow as tf
import pytest

def arrays_bitwise_equal(a, b):
    # Checks both value and sign bit
    return np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))

def test_f():
    numpy_output = np.round([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])

    # CPU output
    with tf.device("/CPU:0"):
        cpu_output = tf.round([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]).numpy()

    # GPU output
    if tf.config.list_physical_devices('GPU'):
        with tf.device("/GPU:0"):
            gpu_output = tf.round([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]).numpy()
    else:
        pytest.skip("No GPU available")

    print("expected numpy", numpy_output)
    print("CPU", cpu_output)
    print("GPU", gpu_output)

    # Check CPU matches NumPy exactly (including signed zero)
    assert arrays_bitwise_equal(numpy_output, cpu_output)

    # Check GPU differs from CPU in sign-bit sense
    assert not arrays_bitwise_equal(cpu_output, gpu_output)
