import tensorflow as tf
import numpy as np

# Create minimal spectrogram with insufficient frequency resolution
# Shape [1, 1, 2] means only 2 frequency bins available
spectrogram = tf.constant([[[1.0, 2.0]]], dtype=tf.float32)

# Parameters that cause the segmentation fault
sample_rate = tf.constant(1744830464, dtype=tf.int32)  # Very large sample rate
lower_freq = float('nan')  # NaN lower frequency limit
upper_freq = 0.498889
filterbank_channels = 18  # Requesting 18 channels but only 2 freq bins available

# This crashes with segmentation fault after logging the error
result = tf.raw_ops.Mfcc(
    spectrogram=spectrogram,
    sample_rate=sample_rate,
    upper_frequency_limit=upper_freq,
    lower_frequency_limit=lower_freq,
    filterbank_channel_count=filterbank_channels,
    dct_coefficient_count=4
)