import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
padding = "SAME"
data_format = "NCHW_VECT_C"
input = tf.random.uniform([1, 1, 16, 2, 1], dtype=tf.float32)
ksize = [1, 1, 1, 1]
strides = [1, 1, 1, 1]
res = tf.raw_ops.MaxPoolV2(
            padding=padding,
            data_format=data_format,
            input=input,
            ksize=ksize,
            strides=strides,
        )
print(res)