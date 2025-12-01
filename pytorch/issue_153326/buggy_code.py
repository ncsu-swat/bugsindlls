import torch

print(torch.__version__)

tensor = torch.tensor([])

torch.choose_qparams_optimized(
    tensor,
    0,
    1,
    1152922350715404288,
    1
)