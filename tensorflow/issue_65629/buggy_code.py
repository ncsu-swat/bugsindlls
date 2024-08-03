import tensorflow as tf

input_data = {'image': tf.random.normal(shape=(2, 2, 2, 2), dtype=tf.float64),
  'boxes': tf.constant([], dtype=tf.float32),
  'box_ind': tf.constant([], dtype=tf.int32),
  'crop_size': tf.constant([1, 2], dtype=tf.int32)
}

result = tf.raw_ops.CropAndResize(**input_data)

print(result)