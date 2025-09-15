import tensorflow as tf
with tf.device("GPU:0"):
    orig_input = tf.random.uniform([16, 28, 30, 10, 0], dtype=tf.float32)
    orig_output = tf.random.uniform([16, 2, 1, 2, 0], dtype=tf.float32)
    grad = tf.random.uniform([16, 2, 1, 2, 0], dtype=tf.float32)
    ksize = [1, 22, 12, 17, 1]
    strides = [1, 16, 30, 7, 1]
    padding = "SAME"
    data_format = "NDHWC"
    res = tf.raw_ops.MaxPool3DGrad(
        orig_input=orig_input,
        orig_output=orig_output,
        grad=grad,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )