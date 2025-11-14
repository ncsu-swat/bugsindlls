import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)

# Create resource variables
var = tf.Variable(np.random.random((6, 6)).astype(np.float32), dtype=tf.bfloat16)
accum = tf.Variable(np.zeros((6, 6)).astype(np.float32), dtype=tf.bfloat16)

# Create tensors with problematic configurations
lr_tensor = tf.constant(0.0, dtype=tf.bfloat16)
grad_tensor = tf.constant(np.zeros(10), dtype=tf.bfloat16)  # Shape [10] - mismatch
indices_tensor = tf.constant([0, 0, 0, 1, 1, 2, 3, 4], dtype=tf.int32)  # 8 indices
momentum_tensor = tf.constant(0.9, dtype=tf.bfloat16)

# This crashes the process
result = tf.raw_ops.ResourceSparseApplyMomentum(
    var=var.handle,
    accum=accum.handle,
    lr=lr_tensor,
    grad=grad_tensor,
    indices=indices_tensor,
    momentum=momentum_tensor,
    use_locking=True,
    use_nesterov=True
)