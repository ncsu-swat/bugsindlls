import pytest
import tensorflow as tf
import numpy as np
import torch

def test_tf_lrn_cpu_gpu_difference():
    
    print(tf.__version__)   # 2.20.0-dev20250715
    print(torch.__version__)  # 2.7.1+cu126
    
    a = tf.random.uniform(shape=(1, 1, 3, 1), minval=0, maxval=1, dtype=tf.float32, seed=100)
    depth_radius = 5
    alpha = 10
    beta = -1  
    output_torch_cpu = torch.nn.functional.local_response_norm(torch.tensor(a.numpy()), size=depth_radius, alpha=alpha, beta=beta)
    print("\nTorch output [cpu]:", output_torch_cpu)

    output_torch_gpu = torch.nn.functional.local_response_norm(torch.tensor(a.numpy()).cuda(), size=depth_radius, alpha=alpha, beta=beta)
    print("\nTorch output [gpu]:", output_torch_gpu)
    
    assert np.array_equal(output_torch_cpu.numpy(), output_torch_gpu.cpu().numpy()), "Torch CPU and GPU outputs match!"
    
    with tf.device('/cpu:0'):
        output_cpu = tf.nn.local_response_normalization(a, depth_radius=depth_radius, alpha=alpha, beta=beta)
        print("\nTensorflow output [cpu]:", output_cpu)

    with pytest.raises(tf.errors.InvalidArgumentError) as e_info:
        with tf.device('/gpu:0'):
            output_gpu = tf.nn.local_response_normalization(a, depth_radius=depth_radius, alpha=alpha, beta=beta)
            print("\nTensorflow output [gpu]:", output_gpu)
        print(f'{e_info.type.__name__}: {e_info.value}')