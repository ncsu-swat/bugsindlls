import tensorflow as tf
import numpy as np
import os

# oneDNN may be involved but crash occurs regardless
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Grads tensor - shape [2,3,3,2] with extreme float values
grads_data = np.array([
    [[[-1.14306452e+18, -4.13623095e-14], [2.83210481e+20, 1.0], [1.0, 1.0]],
     [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
     [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]],
    [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
     [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], 
     [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]
], dtype=np.float32)
grads = tf.constant(grads_data, dtype=tf.float32)

# Image tensor - shape [1,5,5,2] with uint16 values
image_data = np.random.randint(0, 65535, size=(1, 5, 5, 2), dtype=np.uint16)
image = tf.constant(image_data, dtype=tf.uint16)

# CRITICAL: Boxes tensor with NaN values - this causes the crash
boxes_data = np.array([
    [np.nan, 1.0, 1.0, 1.0],  # NaN in first coordinate
    [0.0, 0.0, 1.0, 1.0]      # Valid box
], dtype=np.float32)
boxes = tf.constant(boxes_data, dtype=tf.float32)

# Box indices tensor
box_ind = tf.constant([0, 0], dtype=tf.int32)

# This call causes segmentation fault due to NaN in boxes
result = tf.raw_ops.CropAndResizeGradBoxes(
    grads=grads,
    image=image,
    boxes=boxes,
    box_ind=box_ind
)