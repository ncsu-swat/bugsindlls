import torch
input = torch.rand([9, 5, 13, 0], dtype=torch.float32)
weight = torch.rand([7, 0, 11], dtype=torch.float32)
kernel_size = [1, 2, 3]
bias = torch.rand([10, 10, 2, 6], dtype=torch.float32)
stride = 1
torch._C._nn.slow_conv3d(
    input=input,
    weight=weight,
    kernel_size=kernel_size,
    bias=bias,
    stride=1,
)