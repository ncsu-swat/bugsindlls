import tensorflow as tf
import numpy as np
import sys
def test_f():
    print('Using tensorflow', tf.__version__)
    print('Using python', sys.version)
    data = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [4, 3, 2, 1],
                    [8, 7, 6, 5]], dtype=np.float32)

    segment_ids = tf.constant([1, 2, 3, 4], dtype=tf.int32)
    data = tf.constant(data, dtype=tf.float32)

    # GPU
    with tf.device('/GPU:0'):
        gpu_output_raw = tf.raw_ops.SegmentMax(data=data, segment_ids=segment_ids)
        gpu_output_math = tf.math.segment_max(data=data, segment_ids=segment_ids)

    # CPU
    with tf.device('/CPU:0'):
        cpu_output_raw = tf.raw_ops.SegmentMax(data=data, segment_ids=segment_ids)
        cpu_output_math = tf.math.segment_max(data=data, segment_ids=segment_ids)

    # Convert to numpy
    gpu_raw = gpu_output_raw.numpy()
    gpu_math = gpu_output_math.numpy()
    cpu_raw = cpu_output_raw.numpy()
    cpu_math = cpu_output_math.numpy()

    print("GPU raw:\n", gpu_raw)
    print("GPU math:\n", gpu_math)
    print("CPU raw:\n", cpu_raw)
    print("CPU math:\n", cpu_math)

    # Assert they are NOT equal
    assert not np.array_equal(cpu_raw, gpu_raw), "Expected CPU and GPU raw outputs to differ"
    assert not np.array_equal(cpu_math, gpu_math), "Expected CPU and GPU math outputs to differ"
