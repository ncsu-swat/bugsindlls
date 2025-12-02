import torch
input = torch.rand([], dtype=torch.float32).cuda()
indices = []
values = torch.rand([5], dtype=torch.float32).cuda()
accumulate = True
res = torch.index_put_(
    input=input,
    indices=indices,
    values=values,
    accumulate=accumulate,
)