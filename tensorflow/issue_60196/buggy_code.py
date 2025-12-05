import tensorflow as tf
with tf.device("GPU:0"):
    shape = [94, 47, 76, 127, 90]
    seed = tf.saturate_cast(tf.random.uniform([2], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.int32)
    alpha = tf.saturate_cast(tf.random.uniform([], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.half)
    res = tf.raw_ops.StatelessRandomGammaV2(
        shape=shape,
        seed=seed,
        alpha=alpha,
    )
    print("Success")