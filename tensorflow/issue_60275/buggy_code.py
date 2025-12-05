import tensorflow as tf
with tf.device("CPU"):
    input = tf.complex(tf.random.uniform([0, 9, 13, 15, 13, 8440370290997831992], dtype=tf.float32, minval=-1024, maxval=1024),tf.random.uniform([0, 9, 13, 15, 13, 8440370290997831992], dtype=tf.float32, minval=-1024, maxval=1024))
    res = tf.raw_ops.MatrixLogarithm(
        input=input,
    )