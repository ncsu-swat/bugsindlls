import numpy as np
import tensorflow as tf

rng = np.random.default_rng(24)

with tf.device('/GPU:0'):
    input_tensor = tf.constant(rng.uniform(-10.9999152216621, 6.999802457840147,(8, 10, 13, 12, 2, 4)), dtype=tf.float64)
    pinverse_tensor = tf.linalg.pinv(input_tensor)