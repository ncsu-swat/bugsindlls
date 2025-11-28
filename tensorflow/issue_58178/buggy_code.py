import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        with tf.device("GPU:0"):
            binary_output = False
            input = tf.saturate_cast(tf.random.uniform([0, 5], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int64)
            size = 340
            weights = tf.random.uniform([13, 0], dtype=tf.float32)
            res = tf.raw_ops.DenseBincount(
                binary_output=binary_output,
                input=input,
                size=size,
                weights=weights,
            )
    except:
        pass