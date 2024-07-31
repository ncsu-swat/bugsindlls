import tensorflow as tf


input_dict = {
    'params': tf.constant([[-8., -8.]], dtype=tf.float32),
    'indices': tf.constant(-623719424, dtype=tf.int32),
    'axis': tf.constant(9223372036854775807, dtype=tf.int64)
}


result = tf.raw_ops.GatherV2(**input_dict)


