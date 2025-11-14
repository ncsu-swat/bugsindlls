import tensorflow as tf

def reproduce_crash():
    # Large int64 values that trigger the crash
    values = tf.constant([
        [-9114860691027166365, 9114861777597660793],
        [9114861777597660798, -9114860691027166365],
        [9114861777597660793, 9114861777597660798],
        [9114860691027166365, -9114861777597660793]
    ], dtype=tf.int64)
    
    # Value range
    value_range = tf.constant([9114861777597660672, 9114861777597660798], dtype=tf.int64)
    
    # Number of bins
    nbins = tf.constant(35, dtype=tf.int32)
    
    # This causes segmentation fault
    result = tf.raw_ops.HistogramFixedWidth(
        values=values,
        value_range=value_range,
        nbins=nbins,
        dtype=tf.int32
    )
    return result

# Run this to reproduce the crash
reproduce_crash()