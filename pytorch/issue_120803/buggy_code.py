import torch
inp = torch.rand([0,0])
running_mean = torch.rand([1])
running_var = torch.rand([1])
torch.batch_norm_update_stats(input=inp, momentum=0.0, running_mean=running_mean, running_var = running_var)