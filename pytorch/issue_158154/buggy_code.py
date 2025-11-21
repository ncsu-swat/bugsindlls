import torch

input_tensor = torch.ones(1, 2, 3, 4)
groups = 3

output = torch.native_channel_shuffle(input_tensor, groups)

# Floating point exception (core dumped)