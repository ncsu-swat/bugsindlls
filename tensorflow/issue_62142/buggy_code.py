import tensorflow as tf 
args = {'data_format': 'NHWC', 
 'dilations': [1], 
 'explicit_paddings': [2], 
 'filter_sizes': [3], 
 'input': tf.random.normal([2, 7]), 
 'out_backprop': tf.random.normal([10]), 
 'padding': 'VALID', 'strides': [3], 
 'use_cudnn_on_gpu': True}
res = tf.raw_ops.Conv2DBackpropFilter(**args)
print(res)