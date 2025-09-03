import torch
input = torch.rand([0, 2**20, 4001423662007321682], dtype=torch.float32)
res = torch.clone(
    input=input,
)