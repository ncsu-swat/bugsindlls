import torch
torch._C._nn.slow_conv3d(torch.rand([9]), bias=torch.rand([1,1]), kernel_size=[1], 
                         padding =[1], stride=[], weight=torch.rand([4]))