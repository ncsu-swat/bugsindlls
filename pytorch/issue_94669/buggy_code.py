import torch
input = torch.rand([14, 0, 2423179390303677969, 10, 6], dtype=torch.float32)
some = False
compute_uv = True
res = torch.svd(
    input=input,
    some=some,
    compute_uv=compute_uv,
)