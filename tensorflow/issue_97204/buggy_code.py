import numpy as np
import tensorflow as tf

# Set seed for reproducibility
np.random.seed(202)

# Generate input tensors with non-broadcastable shapes
# x.shape = (4, 1)
# y.shape = (1, 28, 2, 3, 2)
x = np.random.uniform(-32767., 127., size=(4, 1)).astype(np.float32)
y = np.random.uniform(0., 89., size=(1, 28, 2, 3, 2)).astype(np.float32)

# Convert to TensorFlow tensors
x_tensor = tf.constant(x, dtype=tf.float32)
y_tensor = tf.constant(y, dtype=tf.float32)

# --- CPU Execution ---
# This runs without error and produces a misleading result
try:
     with tf.device("/CPU:0"):
         result_cpu = tf.raw_ops.NotEqual(
             x=x_tensor,
             y=y_tensor,
             incompatible_shape_error=False,
             name="selu_cpu",
         )
     print("CPU Result:", result_cpu)
except Exception as e:
     print("CPU Error:", e)


# --- GPU Execution ---
# This correctly fails with an InvalidArgumentError
try:
     with tf.device("/GPU:0"):
         result_gpu = tf.raw_ops.NotEqual(
             x=x_tensor,
             y=y_tensor,
             incompatible_shape_error=False,
             name="selu_gpu",
         )
     print("GPU Result:", result_gpu)
except Exception as e:
     print("\nGPU Error:", e)