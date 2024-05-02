import tensorflow as tf
import sys

print('Using tensorflow', tf.__version__)
print('Using python', sys.version)

tf.raw_ops.RecordInput(
    file_pattern="a",
    file_random_seed=301,
    file_shuffle_shift_ratio=0,
    file_buffer_size=10000,
    file_parallelism=16,
    batch_size=-1,
    compression_type='',
    name=None)
