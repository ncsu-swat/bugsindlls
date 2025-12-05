import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        ksize_0 = 1
        ksize_1 = 2
        ksize_2 = 4
        ksize_3 = 1
        ksize = [ksize_0, ksize_1, ksize_2, ksize_3, ]
        strides_0 = 1
        strides_1 = 2
        strides_2 = 1
        strides_3 = 16
        strides = [strides_0, strides_1, strides_2, strides_3, ]
        padding = "SAME"
        include_batch_in_index = False
        input = tf.saturate_cast(tf.random.uniform([16, 16, 1, 1], minval=0, maxval=64, dtype=tf.int64), dtype=tf.uint64)
        grad = tf.saturate_cast(tf.random.uniform([16, 8, 1, 1], minval=0, maxval=64, dtype=tf.int64), dtype=tf.uint64)
        argmax = tf.saturate_cast(tf.random.uniform([16, 8, 1, 1], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int64)
        res = tf.raw_ops.MaxPoolGradWithArgmax(
            ksize=ksize,
            strides=strides,
            padding=padding,
            include_batch_in_index=include_batch_in_index,
            input=input,
            grad=grad,
            argmax=argmax,
        )
    except:
        pass