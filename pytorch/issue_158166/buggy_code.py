import torch
import numpy as np

rng = np.random.default_rng(272)

input_tensor = torch.tensor(rng.uniform(-1, 0, size=(1, 9, 1, 1)), dtype=torch.float64)
norm_type = 2
kernel_size = 1
stride = np.iinfo(np.int32).max + 1
ceil_mode = False

output_cpu = torch.nn.LPPool2d(norm_type=norm_type, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)(input_tensor)
# No error
output_gpu = torch.nn.LPPool2d(norm_type=norm_type, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode).cuda()(input_tensor.cuda())
# RuntimeError: integer out of range