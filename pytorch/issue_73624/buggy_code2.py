import torch
kernel_size = [3, 2, 2]
output_size = [0, 1, 1]
input = torch.rand([20, 16, 50, 44, 31], dtype=torch.float32)

torch.nn.FractionalMaxPool3d(kernel_size,output_size=output_size)(input)
# segmentation fault (core dumped)