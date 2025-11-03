import tensorflow as tf
import numpy as np

# Create input tensors matching the crash scenario
orig_input_shape = tf.constant([8, 10, 7, 1, 9], dtype=tf.int32)

# Grad tensor with extremely small bfloat16 values that trigger the crash
grad_data = np.full((8, 10, 7, 1, 9), 9.18355e-41, dtype=np.float32)  
grad = tf.constant(grad_data, dtype=tf.bfloat16)

# Pool parameters
ksize = [1, 2, 2, 2, 1]      # [batch, depth, height, width, channels]
strides = [1, 1, 1, 1, 1]    # [batch, depth, height, width, channels]
padding = "VALID"
data_format = "NDHWC"

# This call causes the floating point exception and core dump
result = tf.raw_ops.AvgPool3DGrad(
    orig_input_shape=orig_input_shape,
    grad=grad,
    ksize=ksize,
    strides=strides,
    padding=padding,
    data_format=data_format
)