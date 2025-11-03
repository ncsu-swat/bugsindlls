import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.version.VERSION}")
print(f"oneDNN enabled: {'TF_ENABLE_ONEDNN_OPTS' not in os.environ or os.environ.get('TF_ENABLE_ONEDNN_OPTS', '1') == '1'}")

# Create input tensor
input_tensor = tf.random.normal([10, 1, 1, 9], dtype=tf.float32)

try:
    # This crashes with oneDNN enabled, works with oneDNN disabled
    result = tf.raw_ops.MaxPool(
        input=input_tensor,
        ksize=[1, 1, 1, 1],
        strides=[1, 1, 1, 1], 
        padding="SAME",
        data_format="NCHW_VECT_C"  # Unsupported by MklNativeMaxPool
    )
    print("No crash occurred")
except Exception as e:
    print(f"Exception (expected): {e}")