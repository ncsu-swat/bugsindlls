import torch
import pytest
import numpy as np

def test_f():
    torch.manual_seed(420)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 5])
    loss = torch.nn.MultiMarginLoss()
    gpu_output =loss(input.cuda(), target.cuda())
    print(gpu_output)

    with pytest.raises(Exception) as e_info:
        cpu_output = loss(input, target)
        print(cpu_output)
    print(f'{e_info.type.__name__}: {e_info.value}')