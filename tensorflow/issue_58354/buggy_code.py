import tensorflow as tf
import numpy as np
print(tf.__version__)

indices_data = np.array([[4, 3, 1, 7], [1, 5, 1, 0]])
updates_data = np.array([9, 10, 11, 12])
indices = tf.Variable(indices_data, dtype=tf.int32)
updates = tf.Variable(updates_data, dtype=tf.int32)
stitch = tf.dynamic_stitch(indices, updates)
print(stitch)