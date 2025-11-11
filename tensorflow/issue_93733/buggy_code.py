import tensorflow as tf
import numpy as np
rng = np.random.default_rng(40)

tf.config.experimental.enable_op_determinism()

with tf.device("/CPU:0"):
    input_tensor = tf.constant(rng.uniform(-3.9808033, -3.0141976, (1, 4, 4, 5)), dtype=tf.float32)
    weight_tensor = tf.constant(rng.uniform(2.0, 2.0, (4, 4, 4, 4)), dtype=tf.float32)
    stride = 2
    padding = 3
    output_padding = 0
    groups = 1
    dilation = 1
    
    input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])
    weight_tensor = tf.transpose(weight_tensor, [2, 3, 1, 0])

    batch_size = tf.shape(input_tensor)[0]
    input_height = input_tensor.shape[1]
    input_width = input_tensor.shape[2]
    filter_height = weight_tensor.shape[0]
    filter_width = weight_tensor.shape[1]
    
    out_channels = weight_tensor.shape[3]
    
    h_out = (input_height - 1) * stride - 2 * padding + dilation * (filter_height - 1) + 1 + output_padding
    w_out = (input_width - 1) * stride - 2 * padding + dilation * (filter_width - 1) + 1 + output_padding
    output_shape = [batch_size, h_out, w_out, out_channels]

    output = tf.nn.conv2d_transpose(
        input_tensor, weight_tensor, output_shape=output_shape, strides=[1, stride, stride, 1],
        padding='VALID' if padding == 0 else 'SAME', dilations=[1, dilation, dilation, 1]
    )