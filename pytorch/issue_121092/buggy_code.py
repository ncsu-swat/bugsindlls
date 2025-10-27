import torch
a = torch.rand([0,1,1,1])
torch.native_channel_shuffle(a, groups=1)