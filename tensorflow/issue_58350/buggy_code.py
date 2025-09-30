import tensorflow as tf
import numpy as np
true_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_true = 2
num_sampled = 4
unique = True
range_max = 10
distortion = 1.0
num_reserved_ids = 0
num_shards = 1
shard = 0
unigrams = [0.1, 0.8, 0.1]
seed = None
sampler = tf.random.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, distortion=distortion, num_reserved_ids=num_reserved_ids, num_shards=num_shards, shard=shard, unigrams=unigrams, seed=seed)