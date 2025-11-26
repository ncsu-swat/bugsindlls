import tensorflow as tf
l = tf.random.uniform([8, 0, 2**60, 8], dtype=tf.float64, minval=-2**50, maxval=2**50)
grad = tf.random.uniform([], dtype=tf.float64, minval=-2**50, maxval=2**50)
res = tf.raw_ops.CholeskyGrad(
    l=l,
    grad=grad,
)