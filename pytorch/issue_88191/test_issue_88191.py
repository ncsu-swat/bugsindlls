import torch
import pytest
import numpy as np

def test_f():
    m = torch.nn.RReLU(lower=0.3, upper=0.2)
    x = torch.tensor([-1., -1, -1, -1]).cuda()
    cpu_output = m(x)
    print(cpu_output)

    with pytest.raises(Exception) as e_info:
        y = torch.tensor([-1., -1, -1, -1])
        gpu_output = m(y)

    print(f'{e_info.type.__name__}: {e_info.value}')