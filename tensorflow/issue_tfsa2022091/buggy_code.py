import tensorflow as tf
import numpy as np

value = np.ones([1, 1, 1, 1])
ksize = [1, 1e20, 1, 1]
strides = [1, 1, 1, 1]
padding = 'SAME'
data_format = 'NHWC'

tf.raw_ops.AvgPool(value=value, ksize=ksize, strides=strides, padding=padding, data_format=data_format)