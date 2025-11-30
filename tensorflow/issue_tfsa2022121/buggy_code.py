import tensorflow as tf
import numpy as np
input_sizes = [3, 1, 1, 2]
filter = np.ones([1, 3, 2, 3])
out_backprop = np.ones([3, 1, 0, 3])
strides = [1, 1, 2, 1]
padding = 'VALID'

tf.raw_ops.Conv2DBackpropInput(
   input_sizes = input_sizes,
   filter = filter,
   out_backprop = out_backprop,
   strides = strides,
   padding = padding
)