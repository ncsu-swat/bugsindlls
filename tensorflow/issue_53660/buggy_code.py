import sys
import tensorflow as tf

print('Using tensorflow', tf.__version__)
print('Using python', sys.version)

data = tf.random.uniform([1, 32, 32], dtype=tf.float32)
axis = [1, 2]
x = tf.sparse.from_dense(data)
result = tf.sparse.split(x,3, axis=axis)
