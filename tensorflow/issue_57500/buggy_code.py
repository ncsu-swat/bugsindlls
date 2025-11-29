import tensorflow as tf
with tf.device("CPU"):
    input = tf.random.uniform([30, 19, 4, 19, 17], dtype=tf.float32)
    ksize =[1, 13, 3, 20, 1]
    strides = [1, 14, 4, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    res = tf.raw_ops.AvgPool3D(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )