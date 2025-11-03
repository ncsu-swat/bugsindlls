import tensorflow as tf

with tf.device('/cpu:0'):
    try:
        grad = tf.constant([0, 0, 0, 0], dtype=tf.float64, shape=[4])
        indices = tf.constant([-1], dtype=tf.int64, shape=[1])
        segment_ids = tf.constant([-1], dtype=tf.int64, shape=[1])
        dense_output_dim0 = tf.constant([1], dtype=tf.int32, shape=[])

        tf.raw_ops.SparseSegmentSumGradV2(
            grad=grad,
            indices=indices,
            segment_ids=segment_ids,
            dense_output_dim0=dense_output_dim0,
            name=None
        )
        print("SparseSegmentSumGradV2 executed successfully on CPU")
    except Exception as e:
        print(f"Exception on CPU: {e}")
        
with tf.device('/gpu:0'):
    try:
        grad = tf.constant([0, 0, 0, 0], dtype=tf.float64, shape=[4])
        indices = tf.constant([-1], dtype=tf.int64, shape=[1])
        segment_ids = tf.constant([-1], dtype=tf.int64, shape=[1])
        dense_output_dim0 = tf.constant([1], dtype=tf.int32, shape=[])

        tf.raw_ops.SparseSegmentSumGradV2(
            grad=grad,
            indices=indices,
            segment_ids=segment_ids,
            dense_output_dim0=dense_output_dim0,
            name=None
        )
        print("SparseSegmentSumGradV2 executed successfully on GPU")
    except Exception as e:
        print(f"Exception on GPU: {e}")