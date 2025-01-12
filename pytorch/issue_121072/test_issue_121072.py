import torch
from torch.nn.functional import interpolate
import pytest

def get_grads(dtype, size):
    input = torch.randn((2, 2, 1, 1), dtype=dtype, requires_grad=True).cuda()

    output = interpolate(input, size=size, mode='bicubic', align_corners=True)

    grad_outputs = [torch.ones_like(output)]
    grads = torch.autograd.grad([output], [input], grad_outputs)
    return grads[0]

def test_f():
    issue_no = '121072'
    print('Pytorch issue no.', issue_no)

    print(get_grads(torch.float32, (128, 128)).flatten())
    print(get_grads(torch.float16, (128, 128)).flatten())
    assert torch.all(get_grads(torch.float16, (128, 128)).flatten() == 2048) # It appears that the maximum value of the gradient is capped at 2048 for the float16 data type

    print(get_grads(torch.float32, (64, 64)).flatten())
    print(get_grads(torch.float16, (64, 64)).flatten())
    assert torch.all(get_grads(torch.float16, (64, 64)).flatten() == 2048) # It appears that the maximum value of the gradient is capped at 2048 for the float16 data type

    print(get_grads(torch.float32, (32, 32)).flatten())
    print(get_grads(torch.float16, (32, 32)).flatten()) # Correct output for float16
