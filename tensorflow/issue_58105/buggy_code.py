import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        with tf.device("GPU:0"):
            window_size = -94
            stride = 83
            magnitude_squared = False
            input = tf.random.uniform([12, 4], dtype=tf.float32)
            res = tf.raw_ops.AudioSpectrogram(
                window_size=window_size,
                stride=stride,
                magnitude_squared=magnitude_squared,
                input=input,
            )
    except:
        pass