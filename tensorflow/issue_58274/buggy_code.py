import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        seed = 0
        seed2 = 0
        shape_0 = 16
        shape_1 = 9
        shape_2 = 117
        shape_3 = 100
        shape_4 = 71
        shape_5 = 18
        shape = [shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, ]
        means = tf.saturate_cast(tf.random.uniform([], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        stdevs = tf.saturate_cast(tf.random.uniform([], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        minvals = tf.saturate_cast(tf.random.uniform([1], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        maxvals = tf.saturate_cast(tf.random.uniform([], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        res = tf.raw_ops.ParameterizedTruncatedNormal(
            seed=seed,
            seed2=seed2,
            shape=shape,
            means=means,
            stdevs=stdevs,
            minvals=minvals,
            maxvals=maxvals,
        )
    except:
        pass