import torch
input = torch.rand([9, 15, 0, 2772747373535906632, 0, 14], dtype=torch.float32)
kernel_size = [ 1 ]
stride = [ 1 ]
padding = [ 1 ]
ceil_mode = False
count_include_pad = False
res = torch.avg_pool1d(
    input=input,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    ceil_mode=ceil_mode,
    count_include_pad=count_include_pad,
)