import tensorflow as tf

with tf.device("CPU"):
    input = tf.complex(tf.random.uniform([0, 1, 7678600331551628182, 15], dtype=tf.float32, minval=-18446744073709551615, maxval=18446744073709551615),tf.random.uniform([0, 1, 7678600331551628182, 15], dtype=tf.float32, minval=-18446744073709551615, maxval=18446744073709551615))
    res = tf.raw_ops.MatrixDeterminant(
        input=input,
    )