import torch
import pytest
import numpy as np

def test_f():
    input = torch.randn(1, 1, 2, 2, requires_grad=True)
    weight = torch.randn(1, 1, 2, 2, requires_grad=True)
    gpu_output = torch.nn.functional.conv_transpose2d(input.cuda(), weight.cuda(), stride=2, padding=2)
    print(gpu_output)

    with pytest.raises(Exception) as e_info:
        cpu_output = torch.nn.functional.conv_transpose2d(input, weight, stride=2, padding=2)
        print(cpu_output)
    print(f'{e_info.type.__name__}: {e_info.value}')