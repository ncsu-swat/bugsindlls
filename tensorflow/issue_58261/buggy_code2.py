import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        x = tf.random.uniform([4, 0, 1], dtype=tf.float32)
        h_prev = tf.random.uniform([4, 1, 1], dtype=tf.float32)
        w_ru = tf.random.uniform([1, 2, 1], dtype=tf.float32)
        w_c = tf.random.uniform([1, 1, 1], dtype=tf.float32)
        b_ru = tf.random.uniform([2], dtype=tf.float32)
        b_c = tf.random.uniform([1], dtype=tf.float32)
        r = tf.random.uniform([4, 1, 1], dtype=tf.float32)
        u = tf.random.uniform([4, 1, 1], dtype=tf.float32)
        c = tf.random.uniform([4, 1, 1], dtype=tf.float32)
        d_h = tf.random.uniform([4, 1, 1], dtype=tf.float32)
        res = tf.raw_ops.GRUBlockCellGrad(
            x=x,
            h_prev=h_prev,
            w_ru=w_ru,
            w_c=w_c,
            b_ru=b_ru,
            b_c=b_c,
            r=r,
            u=u,
            c=c,
            d_h=d_h,
        )
    except:
        pass