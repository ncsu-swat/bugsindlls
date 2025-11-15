import tensorflow as tf

with tf.device('/gpu:0'):
    try:
        value = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.int32, shape=[2,6])
        bias = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32, shape=[6])
        data_format = "NCHW"
        result = tf.raw_ops.BiasAdd(
            value=value,
            bias=bias,
            data_format=data_format,
            name=None
        )
        print(result)
    except Exception as e:
        print(e)