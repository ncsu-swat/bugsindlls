import tensorflow as tf
with tf.device("CPU"):
    Tsplits = tf.int32
    starts = tf.random.uniform([], dtype=tf.float32, minval=-1024, maxval=1024)
    limits = tf.random.uniform([11], dtype=tf.float32, minval=-18446744073709551615, maxval=18446744073709551615)
    deltas = tf.random.uniform([], dtype=tf.float32, minval=-1024, maxval=1024)
    res = tf.ragged.range(
        starts=starts,
        limits=limits,
        deltas=deltas,
        row_splits_dtype=tf.int32,
    )
    print(res)