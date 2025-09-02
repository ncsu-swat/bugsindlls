import tensorflow as tf
with tf.device("GPU:0"):
    orig_input = tf.random.uniform([19, 10, 0, 7, 19], dtype=tf.float32)
    orig_output = tf.random.uniform([19, 4, 0, 4, 19], dtype=tf.float32)
    grad = tf.random.uniform([19, 4, 0, 4, 19], dtype=tf.float32)
    ksize = [1, 16, 10, 31, 1]
    strides = [1, 3, 1, 2, 1]
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