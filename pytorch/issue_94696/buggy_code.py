import torch
input = torch.rand([12, 2, 2, 2, 0, 4634247419717959497], dtype=torch.float32)
kernel_size = [ 2 ]
stride = [ 1 ]
padding = [ 1 ]
dilation = [ 1 ]
ceil_mode = False
res = torch.max_pool1d_with_indices(
    input=input,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    ceil_mode=ceil_mode,
)