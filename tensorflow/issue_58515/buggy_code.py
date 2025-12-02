import tensorflow as tf

with tf.device("/cpu:0"):
  zeros = tf.zeros((3, 3), dtype=tf.float64)
  negs = tf.math.negative(zeros)
  infs = tf.math.reciprocal(negs)
  print(zeros, negs, infs)