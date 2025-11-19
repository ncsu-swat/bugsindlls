# test_issue_97204.py
import pytest
import multiprocessing
import tensorflow as tf
import numpy as np

def run_quant_dequant_bug(queue):
    try:
        # Extreme values test for QuantizeAndDequantizeV3
        input_tensor = tf.constant([
            [[4.3136053036349129e-244, 1.2524328545813582e-21, 7.71590127777328e+26, 1.0, 2.0, 3.0],
             [1e-100, 1e+100, -1e+50, 4.0, 5.0, 6.0]]
        ], dtype=tf.float64)

        input_min = tf.constant(-5.4785109376353583e-282, dtype=tf.float64)
        input_max = tf.constant(1.4455588399771524e+73, dtype=tf.float64)
        num_bits = tf.constant(2, dtype=tf.int32)

        tf.raw_ops.QuantizeAndDequantizeV3(
            input=input_tensor,
            input_min=input_min,
            input_max=input_max,
            num_bits=num_bits,
            signed_input=False,
            range_given=True,
            narrow_range=True,
            axis=-2  # Negative axis triggers crash
        )
        queue.put(False)  # Ran successfully → bug not reproduced
    except Exception:
        queue.put(True)  # Exception → bug reproduced


def run_not_equal_bug(queue):
    try:
        # Set seed for reproducibility
        np.random.seed(202)

        x = np.random.uniform(-32767., 127., size=(4, 1)).astype(np.float32)
        y = np.random.uniform(0., 89., size=(1, 28, 2, 3, 2)).astype(np.float32)

        x_tensor = tf.constant(x, dtype=tf.float32)
        y_tensor = tf.constant(y, dtype=tf.float32)

        # CPU execution
        with tf.device("/CPU:0"):
            tf.raw_ops.NotEqual(
                x=x_tensor,
                y=y_tensor,
                incompatible_shape_error=False,
                name="cpu_test",
            )

        # GPU execution
        with tf.device("/GPU:0"):
            tf.raw_ops.NotEqual(
                x=x_tensor,
                y=y_tensor,
                incompatible_shape_error=False,
                name="gpu_test",
            )
        queue.put(False)  # Ran without error → bug not reproduced
    except Exception:
        queue.put(True)  # Exception → bug reproduced


@pytest.mark.parametrize("bug_func", [run_quant_dequant_bug, run_not_equal_bug])
def test_tensorflow_issue_97204(bug_func):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=bug_func, args=(queue,))
    process.start()
    process.join()

    if process.exitcode != 0:
        # Segfault or crash → bug reproduced
        bug_reproduced = True
    else:
        # Process finished normally, check queue for Python exceptions
        bug_reproduced = queue.get_nowait()

    assert bug_reproduced, "TensorFlow bug 97204 was NOT reproduced!"