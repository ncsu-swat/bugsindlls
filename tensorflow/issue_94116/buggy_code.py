import tensorflow as tf
import numpy as np

# Create input tensor with extreme values
input_tensor = tf.constant([
    [[4.3136053036349129e-244, 1.2524328545813582e-21, 7.71590127777328e+26, 1.0, 2.0, 3.0],
     [1e-100, 1e+100, -1e+50, 4.0, 5.0, 6.0]]
], dtype=tf.float64)  # Shape [1, 2, 6]

# Extreme range values
input_min = tf.constant(-5.4785109376353583e-282, dtype=tf.float64)
input_max = tf.constant(1.4455588399771524e+73, dtype=tf.float64)
num_bits = tf.constant(2, dtype=tf.int32)

# This crashes with segfault
result = tf.raw_ops.QuantizeAndDequantizeV3(
    input=input_tensor,
    input_min=input_min,
    input_max=input_max,
    num_bits=num_bits,
    signed_input=False,
    range_given=True,
    narrow_range=True,
    axis=-2  # Negative axis triggers the crash
)