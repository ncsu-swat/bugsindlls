import tensorflow as tf
import torch

print(tf.__version__)   # 2.20.0-dev20250715
print(torch.__version__)  # 2.7.1+cu126

a = tf.random.uniform(shape=(1, 1, 3, 1), minval=0, maxval=1, dtype=tf.float32, seed=100)
depth_radius = 5
alpha = 10
beta = -1

output_torch_cpu = torch.nn.functional.local_response_norm(torch.tensor(a.numpy()), size=depth_radius, alpha=alpha, beta=beta)
print("\nTorch output [cpu]:", output_torch_cpu)
'''
Torch output [cpu]: tensor([[[[2.7409],
          [0.6304],
          [0.7823]]]])
'''

output_torch_gpu = torch.nn.functional.local_response_norm(torch.tensor(a.numpy()).cuda(), size=depth_radius, alpha=alpha, beta=beta)
print("\nTorch output [gpu]:", output_torch_gpu)
'''
Torch output [gpu]: tensor([[[[2.7409],
          [0.6304],
          [0.7823]]]], device='cuda:0')
'''

with tf.device('/cpu:0'):
    output_cpu = tf.nn.local_response_normalization(a, depth_radius=depth_radius, alpha=alpha, beta=beta)
    print("\nTensorflow output [cpu]:", output_cpu)
'''
Tensorflow output [cpu]: tf.Tensor(
[[[[9.857574 ]
   [1.3554668]
   [1.8604573]]]], shape=(1, 1, 3, 1), dtype=float32)
'''

with tf.device('/gpu:0'):
    output_gpu = tf.nn.local_response_normalization(a, depth_radius=depth_radius, alpha=alpha, beta=beta)
    print("\nTensorflow output [gpu]:", output_gpu)
# InvalidArgumentError: {{function_node __wrapped__LRN_device_/job:localhost/replica:0/task:0/device:GPU:0}} cuDNN requires beta >= 0.01,