import tensorflow as tf
with tf.device("CPU"):
    input = tf.complex(tf.random.uniform([6, 1, 0, 11, 5241981715460094077, 8], dtype=tf.float64, minval=-1024, maxval=1024),tf.random.uniform([6, 1, 0, 11, 5241981715460094077, 8], dtype=tf.float64, minval=-1024, maxval=1024))
    res = tf.raw_ops.Cholesky(
        input=input,
    )