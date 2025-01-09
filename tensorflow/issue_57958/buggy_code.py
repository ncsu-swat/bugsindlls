import tensorflow as tf
import numpy as np
import sys

print('Using tensorflow', tf.__version__)
print('Using python', sys.version)

tf.nn.conv2d_transpose(input=np.ones((2,2,2,2)), output_shape=[114078056, 179835296], strides=[10], filters=1) # tf.nn.conv2d_transpose crash with abort with large output_shape
