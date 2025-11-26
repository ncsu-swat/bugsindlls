import torch
input_data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
output = torch.bitwise_left_shift(input_data, 1, out=torch.Tensor(1, 3, 5, 5))