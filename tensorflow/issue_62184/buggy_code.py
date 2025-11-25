import tensorflow as tf 
print(tf.__version__)
args = {'tensor': tf.random.uniform([4]), 
        'indices': tf.random.uniform([4, 4, 4], 0, 256, dtype=tf.int32), 
        'updates': tf.random.uniform([4])}
res = tf.raw_ops.TensorScatterUpdate(**args)
print(res)