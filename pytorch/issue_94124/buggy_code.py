import torch
input = torch.rand([], dtype=torch.float32)
coefficients = torch.rand([10, 2, 12, 2, 15], dtype=torch.float32)
res = torch._compute_linear_combination(
    input=input,
    coefficients=coefficients,
)