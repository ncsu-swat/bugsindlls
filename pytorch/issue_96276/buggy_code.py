import torch
input = torch.rand([1, 0, 8602409350401326287, 14, 16, 10], dtype=torch.float32)
res = torch.geqrf(
    input=input,
)