import tensorflow as tf
import numpy as np

print(tf.__version__)   # 2.20.0-dev20250716

x = tf.constant([-48], dtype=tf.int64)
y = tf.constant([66], dtype=tf.int64)

with tf.device('/cpu:0'):
    output_cpu = tf.pow(x=x, y=y)
    print("Output on CPU:", output_cpu)  # 0

with tf.device('/gpu:0'):
    output_gpu = tf.pow(x=x, y=y)
    print("Output on GPU:", output_gpu)  # 2304

output_np = np.power(x.numpy(), y.numpy())
print("Output with NumPy:", output_np)  # 0