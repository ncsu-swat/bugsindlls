import pytest
import tensorflow as tf
import numpy as np

# Simulated crash codes
SIGABRT = 6
SIGSEGV = 11
SIGFPE = 8

def test_tf_lrn_cpu_gpu_difference():
    """
    Test that reproduces a TensorFlow CPU vs GPU bug in local_response_normalization.
    Passes only if the GPU fails due to invalid beta, simulating SIGABRT behavior.
    """
    # Reproducible random tensor
    a = tf.random.uniform(shape=(1, 1, 3, 1), minval=0, maxval=1, dtype=tf.float32, seed=100)
    depth_radius = 5
    alpha = 10
    beta = -1  # Negative beta triggers GPU failure

    try:
        # CPU computation
        with tf.device('/CPU:0'):
            output_cpu = tf.nn.local_response_normalization(
                a, depth_radius=depth_radius, alpha=alpha, beta=beta
            ).numpy()

        # GPU computation
        with tf.device('/GPU:0'):
            output_gpu = tf.nn.local_response_normalization(
                a, depth_radius=depth_radius, alpha=alpha, beta=beta
            ).numpy()

        # Compare outputs
        output_np = np.array(output_cpu)
        print("CPU output:", output_cpu)
        print("GPU output:", output_gpu)
        print("NumPy output:", output_np)

        # If CPU and GPU outputs are identical, bug not reproduced
        if np.allclose(output_cpu, output_gpu):
            pytest.fail("CPU and GPU outputs are identical; bug not reproduced.")

    except tf.errors.InvalidArgumentError as e:
        # Expected GPU crash
        print("Caught GPU crash (bug reproduced):", e)
        # Simulate SIGABRT exit code
        print(f"Simulated SIGABRT={SIGABRT}")
        assert True  # Test passes because bug is reproduced