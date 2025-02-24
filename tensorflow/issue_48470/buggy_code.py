import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import tensorflow as tf
import numpy as np
import pytest

print('Using tensorflow', tf.__version__)
print('Using python', sys.version)

input = np.random.rand(2, 8, 8, 8)
x = tf.keras.Input([None, None, 8])
y = tf.keras.layers.Conv2DTranspose(filters=0,kernel_size=3, padding='same', dilation_rate=(1,1))(x)
model = tf.keras.Model(x, y)
z = model(input).numpy()
print(z.mean())
