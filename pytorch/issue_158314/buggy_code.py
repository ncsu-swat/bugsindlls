import torch

input_tensor = torch.ones(32,2, dtype=torch.float64) * -11.
alpha = 324112638312866870.

output_cpu = torch.nn.functional.celu(input_tensor, alpha=alpha)
output_gpu = torch.nn.functional.celu(input_tensor.cuda(), alpha=alpha)

print(output_cpu[0, 0])  # tensor(0., dtype=torch.float64)
print(output_gpu[0, 0])  # tensor(-11.0000, device='cuda:0', dtype=torch.float64)