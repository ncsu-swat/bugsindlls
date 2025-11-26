import tensorflow as tf
import numpy as np
import subprocess
import pytest

SIGABRT_PROCESS_RETURNCODE = -6

def test_f():
    
    print(tf.__version__)   # 2.20.0-dev20250715
    rng = np.random.default_rng(311)

    input_tensor = tf.constant(rng.uniform(-np.finfo(np.float16).max-1, 0., size=(1, 2)), dtype=tf.float16)
    axis = tf.constant(0, dtype=tf.int16)
    output_type = tf.int16
    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        output = tf.math.argmax(input_tensor, axis=axis, output_type=output_type)
    print(f'{e_info.type.__name__}: {e_info.value}')
test_f()