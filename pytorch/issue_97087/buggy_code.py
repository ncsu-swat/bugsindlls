import torch
input = torch.rand([0, 16, 3, 1, 4482056787832389139, 1], dtype=torch.float32)
res = torch.adjoint(
    input=input,
)