import numpy as np
import tensorflow as tf
rng = np.random.default_rng(225)
np_input = rng.uniform(-36.999996, -32.00001, (21, 21, 12, 11, 5, 3)).astype(np.float32)

# try numpy
svd_np = np.linalg.svd(np_input, compute_uv=False)
print("Numpy Successful\n\n")

with tf.device("/GPU:0"):
    input_tensor = tf.constant(np_input)
    svd_tf = tf.linalg.svd(input_tensor, compute_uv=False)