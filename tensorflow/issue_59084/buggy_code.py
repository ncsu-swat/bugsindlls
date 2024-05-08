import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import gen_ragged_conversion_ops
import sys

print('Using tensorflow', tf.__version__)
print('Using python', sys.version)

try:
  arg_0_0_tensor = tf.random.uniform([3], minval=-256, maxval=257, dtype=tf.int64)
  arg_0_0 = tf.identity(arg_0_0_tensor)
  arg_0 = [arg_0_0,]
  arg_1_tensor = tf.random.uniform([], minval=-256, maxval=257, dtype=tf.int32)
  arg_1 = tf.identity(arg_1_tensor)
  arg_2 = True
  arg_3 = None
  out = gen_ragged_conversion_ops.ragged_tensor_to_variant(arg_0,arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))
