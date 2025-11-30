import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        with tf.device("GPU:0"):
            ksize_0 = 1
            ksize_1 = 1
            ksize_2 = 32
            ksize_3 = 32
            ksize_4 = 128
            ksize = [ksize_0, ksize_1, ksize_2, ksize_3, ksize_4, ]
            strides_0 = 1
            strides_1 = 1
            strides_2 = 128
            strides_3 = 128
            strides_4 = 128
            strides = [strides_0, strides_1, strides_2, strides_3, strides_4, ]
            padding = "VALID"
            data_format = "NCDHW"
            orig_input_shape_0 = 111
            orig_input_shape_1 = 24
            orig_input_shape_2 = 43
            orig_input_shape_3 = 77
            orig_input_shape_4 = 89
            orig_input_shape = [orig_input_shape_0, orig_input_shape_1, orig_input_shape_2, orig_input_shape_3, orig_input_shape_4, ]
            grad = tf.random.uniform([0, 4, 1, 1, 1], dtype=tf.float32)
            res = tf.raw_ops.AvgPool3DGrad(
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
                orig_input_shape=orig_input_shape,
                grad=grad,
            )
    except:
        pass