import torch

print(torch.__version__)

tensor = torch.tensor([-9223372036854775808])
divisor = torch.tensor(-2)

result=torch.remainder(tensor, divisor)
print("Result:", result)