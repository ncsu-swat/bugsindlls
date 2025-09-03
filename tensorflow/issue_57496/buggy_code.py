import tensorflow as tf

with tf.device("CPU"):
    res = tf.raw_ops.AdjustHue(
    	images=tf.random.uniform([2, 2, 2, 3]),
        delta=1e100,
    )