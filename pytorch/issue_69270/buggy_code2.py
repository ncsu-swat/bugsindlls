import torch
tensor = torch.rand(torch.Size([2, 2, 4]), dtype=torch.float32)
sections = 0
res1 = torch.vsplit(tensor, sections)