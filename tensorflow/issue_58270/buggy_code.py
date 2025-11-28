import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        forget_bias = 112.66590343649887
        cell_clip = 67.12389445926587
        use_peephole = False
        x = tf.saturate_cast(tf.random.uniform([2, 16], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        cs_prev = tf.saturate_cast(tf.random.uniform([2, 0], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        h_prev = tf.saturate_cast(tf.random.uniform([2, 0], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        w = tf.saturate_cast(tf.random.uniform([16, 0], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        wci = tf.saturate_cast(tf.random.uniform([5], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        wcf = tf.saturate_cast(tf.random.uniform([16], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        wco = tf.saturate_cast(tf.random.uniform([13], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        b = tf.saturate_cast(tf.random.uniform([0], minval=0, maxval=64, dtype=tf.int64), dtype=tf.half)
        res = tf.raw_ops.LSTMBlockCell(
            forget_bias=forget_bias,
            cell_clip=cell_clip,
            use_peephole=use_peephole,
            x=x,
            cs_prev=cs_prev,
            h_prev=h_prev,
            w=w,
            wci=wci,
            wcf=wcf,
            wco=wco,
            b=b,
        )
    except:
        pass