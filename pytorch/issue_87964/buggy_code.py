import torch
import torch.distributed as dist
if True:
    input_tensor = torch.randn(4, 3)
    dist.FileStore('./')