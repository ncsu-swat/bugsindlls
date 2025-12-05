import tensorflow as tf

# The following code is from https://www.tensorflow.org/api_docs/python/tf/data/experimental/CsvDataset
with open('/tmp/my_file0.csv', 'w') as f:
  f.write('abcdefg,4.28E10,5.55E6,12\n')
  f.write('hijklmn,-5.3E14,,2\n')
dataset = tf.data.experimental.CsvDataset(
  "/tmp/my_file0.csv",
  [tf.float32,  # Required field, use dtype or empty tensor
   tf.constant([0.0], dtype=tf.float32),  # Optional field, default to 0.0
   tf.int32,  # Required field, use dtype or empty tensor
  ],
  select_cols=[1,2,3]  # Only parse last three columns
)

# The following operation is causing Check Fail
for e in range(10):
    dataset = dataset.shuffle(1000).repeat().batch(512)