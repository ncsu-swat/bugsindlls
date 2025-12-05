import tensorflow as tf
import numpy as np

# Create input tensor with shape [10,10,2,1] 
input_data = np.random.randint(0, 255, size=(10, 10, 2, 1), dtype=np.uint8)
input_tensor = tf.constant(input_data, dtype=tf.quint8)

# Create filter tensor with shape [3,3,1,3]
filter_data = np.random.randint(0, 255, size=(3, 3, 1, 3), dtype=np.uint8)
filter_tensor = tf.constant(filter_data, dtype=tf.quint8)

# Min/Max tensors for quantization
min_input = tf.constant(0.0, dtype=tf.float32)
max_input = tf.constant(1.0, dtype=tf.float32)
min_filter = tf.constant(0.0, dtype=tf.float32)
max_filter = tf.constant(1.0, dtype=tf.float32)

# This crashes with assertion failure instead of raising proper exception
result = tf.raw_ops.QuantizedConv2D(
    input=input_tensor,
    filter=filter_tensor,
    min_input=min_input,
    max_input=max_input,
    min_filter=min_filter,
    max_filter=max_filter,
    strides=[1, 1, 1, 1],
    padding="VALID",
    dilations=[1, 1, 1, 1]
)

