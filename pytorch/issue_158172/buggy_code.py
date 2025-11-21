import torch

input_tensor = torch.tensor([-2000000], dtype=torch.int64)
other = torch.tensor([-1])
output_cpu = torch.tensor([], dtype=torch.int16)
output_gpu = torch.tensor([], dtype=torch.int16).cuda()

torch.fmin(input_tensor, other, out=output_cpu)
torch.fmin(input_tensor.cuda(), other.cuda(), out=output_gpu)

print(output_cpu) # 31616
print(output_gpu) # -1