import torch

input_tensor = torch.randn(34, 43, dtype=torch.float16)
lambd = 65507.0

out_cpu = torch.nn.functional.softshrink(input_tensor, lambd)
print("No errors on CPU")
out_gpu = torch.nn.functional.softshrink(input_tensor.cuda(), lambd)
# RuntimeError: value cannot be converted to type at::Half without overflow
print("No errors on CUDA")  # does not reach here