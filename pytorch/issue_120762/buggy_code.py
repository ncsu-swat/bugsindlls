import torch

a = torch.tensor([1, 2, 3])
a.flatten(start_dim=0, end_dim=1, out_dim='a')
