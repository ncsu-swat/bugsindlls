import tensorflow as tf 
args = {'value':  tf.random.uniform([8, 2], 0, 256, dtype=tf.int32), 
        'bias':  tf.random.uniform([5],0, 256, dtype=tf.int32), 
        'data_format': 'NCHW'}
res = tf.raw_ops.BiasAdd(**args)
print(res)