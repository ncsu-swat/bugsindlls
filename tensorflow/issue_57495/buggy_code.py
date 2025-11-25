import tensorflow as tf
import numpy as np

input_sizes = [0, 6, 1, 1, 2]
filter = np.ones([3, 1, 1, 2, 3])
out_backprop = np.ones([0, 4, 1, 1, 3])

strides = [1, 1, 1, 1, 1]
padding = 'VALID'
data_format = 'NDHWC'
dilations = [1, 1, 1, 1, 1]
tf.raw_ops.Conv3DBackpropInputV2(input_sizes=input_sizes,\
    filter=filter,\
    out_backprop=out_backprop,\
    strides=strides,\
    padding=padding,\
    data_format=data_format,\
    dilations=dilations)