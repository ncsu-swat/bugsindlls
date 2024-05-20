import torch

torch._C._nn.thnn_conv2d(torch.rand([1]), bias=torch.rand([3]), kernel_size=[5], padding=[], stride=[9], weight=torch.rand([9]))
