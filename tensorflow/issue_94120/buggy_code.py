import tensorflow as tf
import numpy as np
import os

# CRITICAL: Crash only occurs with oneDNN enabled
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Create input tensor with problematic dimensions and extreme bfloat16 values
input_data = np.full((9, 6, 7, 4, 4), 5.00741786e-32, dtype=np.float32)
input_tensor = tf.constant(input_data, dtype=tf.bfloat16)

# Pooling parameters that create invalid output dimensions
ksize = [1, 4, 5, 5, 1]      # [batch, depth, height, width, channels]
strides = [1, 3, 2, 1, 1]    # [batch, depth, height, width, channels]
padding = "VALID"
data_format = "NDHWC"

# This call triggers division by zero in oneDNN
result = tf.raw_ops.MaxPool3D(
    input=input_tensor,
    ksize=ksize,
    strides=strides,
    padding=padding,
    data_format=data_format
)