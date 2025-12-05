import tensorflow as tf
import numpy as np
with tf.device("CPU"): # also can be triggerred on GPU
   input = np.ones([1, 0, 2, 1])
   filter = np.ones([1, 1, 1, 1])
   strides = ([1, 1, 1, 1])
   padding = "EXPLICIT"
   explicit_paddings = [0 , 0, 1, 1, 1, 1, 0, 0]
   data_format = "NHWC"
   res = tf.raw_ops.Conv2D(
       input=input,
       filter=filter,
       strides=strides,
       padding=padding,
        explicit_paddings=explicit_paddings,
       data_format=data_format,
  )