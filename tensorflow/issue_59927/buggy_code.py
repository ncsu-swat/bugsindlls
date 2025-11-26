import tensorflow as tf
print(tf.__version__)
strides = [1, 128, 128, 1]
padding = "SAME"
dilations = [1, 1, 1, 1]
input = tf.cast(tf.random.uniform([2, 1, 0, 1], minval=0, maxval=64, dtype=tf.int64), dtype=tf.quint8)
filter = tf.cast(tf.random.uniform([1, 1, 1, 1], minval=0, maxval=64, dtype=tf.int64), dtype=tf.quint8)
min_input = tf.random.uniform([], dtype=tf.float32)
max_input = tf.random.uniform([], dtype=tf.float32)
min_filter = tf.random.uniform([], dtype=tf.float32)
max_filter = tf.random.uniform([], dtype=tf.float32)
# res = tf.raw_ops.QuantizedConv2D(
res = tf.compat.v1.nn.quantized_conv2d(
    strides=strides,
    padding=padding,
    dilations=dilations,
    input=input,
    filter=filter,
    min_input=min_input,
    max_input=max_input,
    min_filter=min_filter,
    max_filter=max_filter,
)