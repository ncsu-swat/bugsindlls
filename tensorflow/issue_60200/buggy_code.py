import tensorflow as tf
with tf.device("GPU:0"):
    num_true = 11
    num_sampled = 2
    unique = False
    range_max = 7612169259283414040
    seed = -111
    seed2 = -11
    true_classes = tf.saturate_cast(tf.random.uniform([12, 11], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.int64)
    res = tf.raw_ops.ThreadUnsafeUnigramCandidateSampler(
        num_true=num_true,
        num_sampled=num_sampled,
        unique=unique,
        range_max=range_max,
        seed=seed,
        seed2=seed2,
        true_classes=true_classes,
    )