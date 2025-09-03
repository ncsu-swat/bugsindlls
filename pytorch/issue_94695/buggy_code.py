import torch
input = torch.rand([1, 1], dtype=torch.float32)
other = torch.rand([0, 16, 5294285560538976914], dtype=torch.float32)
res = torch.matmul(
    input=input,
    other=other,
)