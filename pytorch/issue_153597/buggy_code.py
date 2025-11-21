import torch
import numpy as np

rng = np.random.default_rng(1371)

input_tensor = torch.tensor(rng.uniform(7, 7, (7, 8, 1, 1, 8, 7)), dtype=torch.int8)
other_tensor = torch.tensor(rng.uniform(-1, 7, (7, 8, 8, 8, 8, 7)), dtype=torch.float16)

a, b = torch.broadcast_tensors(input_tensor, other_tensor)
# Since floor_divide broadcasts the tensors internally

out_cpu = torch.floor_divide(input_tensor, other_tensor)
out_gpu = torch.floor_divide(input_tensor.cuda(), other_tensor.cuda())

print("Input tensor (Broadcasted):")
print(a[2, 0, 6, 4, 3, 4])  # tensor(7, dtype=torch.int8)
print("Other tensor (Broadcasted):")
print(b[2, 0, 6, 4, 3, 4])  # tensor(-0.0004, dtype=torch.float16)
print("CPU output:")
print(out_cpu[2, 0, 6, 4, 3, 4]) # tensor(-17584., dtype=torch.float16)
print("GPU output:")
print(out_gpu.cpu()[2, 0, 6, 4, 3, 4]) # tensor(-17600., dtype=torch.float16)

torch.testing.assert_close(out_cpu, out_gpu.cpu(), rtol=1e-7, atol=1e-2)
# AssertionError. Greatest absolute difference: 16.0 at index (2, 0, 6, 4, 3, 4)