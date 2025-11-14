import tensorflow as tf
import numpy as np

print(tf.version.GIT_VERSION, tf.version.VERSION)

# Reproduce the crash
input_sizes = tf.constant([5], dtype=tf.int32)  # Invalid: should be 5D for Conv3D
filter_tensor = tf.random.normal([7, 6, 9, 2, 3], dtype=tf.float32)
out_backprop = tf.ones([1, 1, 1, 1, 1], dtype=tf.float32)

# This crashes with segfault
result = tf.raw_ops.Conv3DBackpropInputV2(
    input_sizes=input_sizes,
    filter=filter_tensor,
    out_backprop=out_backprop,
    strides=[1, 1, 1, 1, 1],
    padding="VALID",
    data_format="NCDHW",
    dilations=[1, 1, 1, 1, 1]
)