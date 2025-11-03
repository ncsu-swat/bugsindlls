import tensorflow as tf

with tf.device('/cpu:0'):
    try:
        sparse_indices = tf.constant([1], dtype=tf.int32, shape=[1])
        output_shape = tf.constant([0], dtype=tf.int32, shape=[1])
        sparse_values = tf.constant([0], dtype=tf.uint16, shape=[1])
        default_value = tf.constant([0], dtype=tf.uint16, shape=[])
        validate_indices = False
        result = tf.raw_ops.SparseToDense(
            sparse_indices=sparse_indices,
            output_shape=output_shape,
            sparse_values=sparse_values,
            default_value=default_value,
            validate_indices=validate_indices,
            name=None
        )
        print("SparseToDense executed successfully on CPU")
    except Exception as e:
        print("Exception on CPU:", e)

with tf.device('/gpu:0'):
    try:
        sparse_indices = tf.constant([1], dtype=tf.int32, shape=[1])
        output_shape = tf.constant([0], dtype=tf.int32, shape=[1])
        sparse_values = tf.constant([0], dtype=tf.uint16, shape=[1])
        default_value = tf.constant([0], dtype=tf.uint16, shape=[])
        validate_indices = False
        result = tf.raw_ops.SparseToDense(
            sparse_indices=sparse_indices,
            output_shape=output_shape,
            sparse_values=sparse_values,
            default_value=default_value,
            validate_indices=validate_indices,
            name=None
        )
        print("SparseToDense executed successfully on GPU")
    except Exception as e:
        print("Exception on GPU:", e)