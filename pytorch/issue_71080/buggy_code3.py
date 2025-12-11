import torch
filters = torch.randn(8, 4, 3, 3, 3)
inputs = torch.randn(1, 4, 4, 5, 5)
torch.nn.functional.conv3d(inputs, filters, padding=1, groups=0)
# floating point exception