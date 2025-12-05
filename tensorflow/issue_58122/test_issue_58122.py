import tensorflow as tf
import numpy as np
import pytest
def test_f():
    print('Using tensorflow', tf.__version__)
    input = tf.constant([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])
    depth_radius = 0.1
    bias = 1.0
    alpha = 0.5
    beta = 0.5
    with tf.device("cpu"):
        output = tf.nn.local_response_normalization(input=input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
        print(output)
    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        with tf.device("gpu"):
            output = tf.nn.local_response_normalization(input=input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
            print(output)
    print(f'{e_info.type.__name__}:{e_info.value}')