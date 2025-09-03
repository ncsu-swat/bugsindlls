import torch
input = torch.rand([11, 0, 1500908595704918919, 13, 3], dtype=torch.float32).cuda()
rcond = 1
res = torch.pinverse(
    input=input,
    rcond=rcond,
)