import tensorflow as tf
import numpy as np
import pytest

def test_f():
    print("numpy")
    input_np = np.array([100], dtype=np.complex64)
    exp_numpy = np.exp(input_np)
    abs_numpy = np.abs(exp_numpy)
    print(input_np, exp_numpy, abs_numpy)

    # CPU
    with tf.device("/CPU:0"):
        print("CPU")
        input_cpu = tf.convert_to_tensor([100], tf.complex64)
        exp_cpu = tf.exp(input_cpu)
        abs_cpu = tf.abs(exp_cpu)
        print(input_cpu.numpy(), exp_cpu.numpy(), abs_cpu.numpy())

    # GPU
    if tf.config.list_physical_devices('GPU'):
        with tf.device("/GPU:0"):
            print("GPU")
            input_gpu = tf.convert_to_tensor([100], tf.complex64)
            exp_gpu = tf.exp(input_gpu)
            abs_gpu = tf.abs(exp_gpu)
            print(input_gpu.numpy(), exp_gpu.numpy(), abs_gpu.numpy())
    else:
        pytest.skip("No GPU available for test")

    assert np.array_equal(exp_numpy, exp_cpu.numpy())          # CPU matches NumPy
    assert not np.array_equal(exp_cpu.numpy(), exp_gpu.numpy())        # CPU differs from GPU
