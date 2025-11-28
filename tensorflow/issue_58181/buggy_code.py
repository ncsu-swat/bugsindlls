import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        with tf.device("GPU:0"):
            num_true = 12
            num_sampled = 53
            unique = True
            range_max = 105
            vocab_file = ""
            distortion = 15.07084518523925
            num_reserved_ids = 0
            num_shards = 1
            shard = 0
            unigrams_0 = 45.35209469768597
            unigrams_1 = 73.63755693482213
            unigrams_2 = 51.69763696074902
            unigrams = [unigrams_0, unigrams_1, unigrams_2, ]
            seed = 0
            seed2 = 0
            true_classes = tf.saturate_cast(tf.random.uniform([1, 12], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int64)
            res = tf.raw_ops.FixedUnigramCandidateSampler(
                num_true=num_true,
                num_sampled=num_sampled,
                unique=unique,
                range_max=range_max,
                vocab_file=vocab_file,
                distortion=distortion,
                num_reserved_ids=num_reserved_ids,
                num_shards=num_shards,
                shard=shard,
                unigrams=unigrams,
                seed=seed,
                seed2=seed2,
                true_classes=true_classes,
            )
    except:
        pass