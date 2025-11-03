import tensorflow as tf
import numpy as np

print(tf.__version__)

def run_conv2d_backprop_filter_v2_bug():
    """
    Attempts to demonstrate tf.conv2d_backprop_filter_v2 and triggers the error.
    """

    # --- Define the input tensors based on provided information ---
    # Input Tensor: Tensor<type: float shape: [1,3,28,3]>
    input_values = np.zeros((1, 3, 28, 3), dtype=np.float32)
    input_values[0, 0, 0, 0] = -2.16744329e+12
    input_values[0, 0, 0, 1] = 1.23339599e+30
    input_values[0, 0, 0, 2] = 8.00276139e+12
    input_tensor = tf.constant(input_values, dtype=tf.float32)

    # Filter shape: Tensor<type: int shape: [4]> representing [height, width, in_channels, out_channels]
    filter_shape = tf.constant([3, 1, 3, 1], dtype=tf.int32)

    # Out Backprop Tensor: Tensor<type: float shape: [1,2,10,1]>
    out_backprop_values = np.zeros((1, 2, 10, 1), dtype=np.float32)
    out_backprop_values[0, 0, 0, 0] = 2.54932688e-24
    out_backprop_values[0, 0, 1, 0] = 3.49802447e+36
    out_backprop_values[0, 0, 2, 0] = 7.77881911e+12
    out_backprop = tf.constant(out_backprop_values, dtype=tf.float32)

    # --- Define the convolution parameters ---
    data_format = "NCHW"
    strides = [1, 1, 2, 3] # [N, C, H, W]
    padding = "SAME"
    dilations = [1, 1, 2, 3] # [N, C, H, W]

    print(f"Input Tensor Shape: {input_tensor.shape}")
    print(f"Filter Shape: {filter_shape.numpy()}")
    print(f"Out Backprop Tensor Shape: {out_backprop.shape}")
    print(f"Strides: {strides}")
    print(f"Padding: {padding}")
    print(f"Data Format: {data_format}")
    print(f"Dilations: {dilations}")
    print("-" * 30)

    try:
        grad_filter = tf.conv2d_backprop_filter_v2(
            input=input_tensor,
            filter=filter_shape,
            out_backprop=out_backprop,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilations=dilations
        )
        print("tf.conv2d_backprop_filter_v2 operation completed successfully!")
        print(f"Shape of the gradient of the filter: {grad_filter.shape}")
    except Exception as e:
        print(f"An error occurred during tf.conv2d_backprop_filter_v2: {e}")

if __name__ == "__main__":
    run_conv2d_backprop_filter_v2_bug()