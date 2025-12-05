import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)

# Create tensors with problematic shapes
input_tensor = tf.constant(np.random.random((1, 10, 4)), dtype=tf.float64)
indices_tensor = tf.constant([[[0]]], dtype=tf.int64)  # Malformed: shape (1,1,1) instead of (1,3)
updates_tensor = tf.constant([0.0], dtype=tf.float64)

print("Input Tensor shape:", input_tensor.shape)
print("Indices Tensor shape:", indices_tensor.shape) 
print("Updates Tensor shape:", updates_tensor.shape)

print("--- Testing TensorScatterMax ---")
# This causes a fatal crash
result = tf.raw_ops.TensorScatterMax(
    tensor=input_tensor,
    indices=indices_tensor,
    updates=updates_tensor
)