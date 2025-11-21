import torch
output_gpu = torch.logcumsumexp(torch.tensor(22, dtype=torch.int64).cuda(), 0) # no runtime error
output_gpu = torch.logcumsumexp(torch.tensor([22], dtype=torch.int64).cuda(), 0) # error
output_cpu = torch.logcumsumexp(torch.tensor(22, dtype=torch.int64), 0) # error