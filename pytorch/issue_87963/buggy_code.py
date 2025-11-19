import torch
import torch.nn as nn
print(torch.overrides.has_torch_function(nn.Linear(3, 4)))