import torch
x1 = torch.rand([1, 1, 1], dtype=torch.float32)
x2 = torch.rand([0, 12, 5676819769219801604], dtype=torch.float32)
res = torch._euclidean_dist(
    x1=x1,
    x2=x2,
)