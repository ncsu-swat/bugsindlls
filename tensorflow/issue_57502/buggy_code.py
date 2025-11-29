import tensorflow as tf
with tf.device("GPU:0"):
    orig_input = tf.random.uniform([6, 32, 6, 13], dtype=tf.float32)
    orig_output = tf.random.uniform([6, 2, 0, 13], dtype=tf.float32)
    grad = tf.random.uniform([6, 2, 0, 13], dtype=tf.float32)
    ksize = [1, 13, 8, 1]
    strides = [1, 18, 7, 1]
    padding = "VALID"
    data_format = "NHWC"
    res = tf.raw_ops.MaxPoolGrad(
        orig_input=orig_input,
        orig_output=orig_output,
        grad=grad,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )