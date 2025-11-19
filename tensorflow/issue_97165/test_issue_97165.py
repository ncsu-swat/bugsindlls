import multiprocessing
import tensorflow as tf
import numpy as np
import pytest

def run_gpu_test(queue):
    try:
        input_tensor = tf.constant([[[[1.],[2.]],[[3.],[4.]]]], dtype=tf.float32)
        filter_tensor = tf.constant([[[[1., 2.]]]], dtype=tf.float32)
        strides = [4817177250100823153, 5276955028067489600, -6860092642535747309, -915217906097603218]
        padding = 'VALID'
        data_format = 'NHWC'
        dilations = [1, 1]

        with tf.device('/GPU:0'):
            tf.nn.depthwise_conv2d(
                input_tensor, filter_tensor, strides=strides,
                padding=padding, data_format=data_format,
                dilations=dilations
            ).numpy()
        queue.put(False)  # Bug not reproduced
    except Exception:
        queue.put(True)   # Bug reproduced (Python exception)
    # If segfault occurs, process just dies

def test_depthwise_conv2d_gpu_bug():
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_gpu_test, args=(queue,))
    p.start()
    p.join()

    if p.exitcode != 0:
        # Process crashed → bug reproduced
        print("GPU process crashed: bug reproduced!")
        bug_reproduced = True
    else:
        # Process finished normally → check queue for Python exceptions
        bug_reproduced = queue.get_nowait()

    assert bug_reproduced, "Bug was not reproduced on GPU!"