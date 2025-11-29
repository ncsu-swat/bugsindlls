import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        capacity = 0
        memory_limit = 0
        dtypes_0 = tf.uint64
        dtypes_1 = tf.float32
        dtypes = [dtypes_0, dtypes_1, ]
        container = ""
        shared_name = ""
        key = tf.saturate_cast(tf.random.uniform([6, 14, 4], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int64)
        indices = tf.saturate_cast(tf.random.uniform([2], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int32)
        res = tf.raw_ops.MapPeek(
            capacity=capacity,
            memory_limit=memory_limit,
            dtypes=dtypes,
            container=container,
            shared_name=shared_name,
            key=key,
            indices=indices,
        )
    except:
        pass