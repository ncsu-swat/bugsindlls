import tensorflow as tf

input = tf.constant([248, 225])
indices = tf.constant([[[1]]])
updates = tf.constant([16])
tf.tensor_scatter_nd_update(input, indices, updates)