import torch
kernel_size = [2, 2]
output_size = [0, 1]
input = torch.rand([16, 50, 44, 31], dtype=torch.float32)

torch.nn.FractionalMaxPool2d(kernel_size,output_size=output_size)(input)
print("success")
# segmentation fault (core dumped)