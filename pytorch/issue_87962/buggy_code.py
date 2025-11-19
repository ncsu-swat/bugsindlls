import torch
LU_data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
LU_pivots = torch.tensor([0, 1, 2], dtype=torch.int32)
torch.lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True)