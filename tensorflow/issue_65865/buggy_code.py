import tensorflow as tf
import sys

print('Using tensorflow', tf.__version__)
print('Using python', sys.version)

input_dict = {
    'input_indices': tf.constant([[-1, 1, -1],
                                  [-1, 4, 0]], dtype=tf.int64),
    'input_values': tf.constant([2, -3], dtype=tf.int64),
    'input_shape': tf.constant([0, 2684354560, 666666666666], dtype=tf.int64),
    'reduction_axes': tf.constant([0, 0, 0], dtype=tf.int32)
}

result = tf.raw_ops.SparseReduceMaxSparse(**input_dict)
print(result)
