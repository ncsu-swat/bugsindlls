import torch

print(torch.__version__)

tensor = torch.tensor([0, 0, -9223372036854775808,
        -9223372036854775808, -9223372036854775808, -9223372036854775808,
        -9223372036854775808, -9223372036854775808, 0,
        0, 0])
divisor = torch.tensor(-1)

torch.remainder(tensor, divisor)