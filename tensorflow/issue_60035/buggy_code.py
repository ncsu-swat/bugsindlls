import tensorflow as tf
input = tf.random.uniform([10, 15, 0, 4, 5849693008847355793], dtype=tf.float64, minval=-1024, maxval=1024)
res = tf.raw_ops.MatrixSquareRoot(
    input=input,
)