import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        with tf.device("CPU"):
            strides_0 = 1
            strides_1 = 1
            strides_2 = 128
            strides_3 = 128
            strides = [strides_0, strides_1, strides_2, strides_3, ]
            padding = "VALID"
            explicit_paddings = []
            data_format = "NCHW"
            dilations_0 = 96
            dilations_1 = 25
            dilations_2 = 10
            dilations = [dilations_0, dilations_1, dilations_2, ]
            input = tf.random.uniform([], dtype=tf.float32)
            filter_sizes = tf.saturate_cast(tf.random.uniform([0], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int32)
            out_backprop = tf.random.uniform([16, 2], dtype=tf.float32)
            res = tf.raw_ops.DepthwiseConv2dNativeBackpropFilter(
                strides=strides,
                padding=padding,
                explicit_paddings=explicit_paddings,
                data_format=data_format,
                dilations=dilations,
                input=input,
                filter_sizes=filter_sizes,
                out_backprop=out_backprop,
            )
    except:
        pass