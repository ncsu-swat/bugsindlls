import tensorflow as tf

print(tf.__version__)   # 2.20.0-dev20250716

input_tensor = tf.constant([[[[1.],[2.]],[[3.],[4.]]]], dtype=tf.float32)
filter_tensor = tf.constant([[[[1., 2.]]]], dtype=tf.float32)
strides = [4817177250100823153, 5276955028067489600, -6860092642535747309, -915217906097603218]
padding = 'VALID'
data_format = 'NHWC'
dilations = [1, 1]

with tf.device('/cpu:0'):
    output_cpu = tf.nn.depthwise_conv2d(input_tensor, filter_tensor, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name='depthwise_conv2d_1')