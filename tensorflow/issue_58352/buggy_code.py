import tensorflow as tf
input_splits = [[0, 2, 3, 5, 6], [0, 1, 2], [0, 1, 2, 3, 4, 15]]
input_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
output_tensor = tf.raw_ops.RaggedTensorToVariant(rt_nested_splits=input_splits, rt_dense_values=input_values, batched_input=True)
print(output_tensor)