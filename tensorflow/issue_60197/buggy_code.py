import tensorflow as tf
with tf.device("GPU:0"):
    num_true = 13
    num_sampled = 48
    unique = True
    range_max = 3031324185113192368
    seed = 93
    seed2 = 11
    true_classes = tf.saturate_cast(tf.random.uniform([14, 13], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.int64)
    res = tf.raw_ops.LearnedUnigramCandidateSampler(
        num_true=num_true,
        num_sampled=num_sampled,
        unique=unique,
        range_max=range_max,
        seed=seed,
        seed2=seed2,
        true_classes=true_classes,
    )