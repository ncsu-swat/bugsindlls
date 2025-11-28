import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        with tf.device("GPU:0"):
            forget_bias = -121.22699269620765
            cell_clip = -106.82307555235684
            use_peephole = False
            seq_len_max = tf.saturate_cast(tf.random.uniform([13, 11, 0], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int64)
            x = tf.random.uniform([1, 3, 15], dtype=tf.float32)
            cs_prev = tf.random.uniform([3, 0], dtype=tf.float32)
            h_prev = tf.random.uniform([3, 0], dtype=tf.float32)
            w = tf.random.uniform([15, 0], dtype=tf.float32)
            wci = tf.random.uniform([0], dtype=tf.float32)
            wcf = tf.random.uniform([0], dtype=tf.float32)
            wco = tf.random.uniform([0], dtype=tf.float32)
            b = tf.random.uniform([0], dtype=tf.float32)
            res = tf.raw_ops.BlockLSTM(
                forget_bias=forget_bias,
                cell_clip=cell_clip,
                use_peephole=use_peephole,
                seq_len_max=seq_len_max,
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