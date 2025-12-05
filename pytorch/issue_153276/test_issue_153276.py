import torch
import numpy as np
import pytest

def test_f():
    torch.backends.cudnn.deterministic = True

    rng = np.random.default_rng(1146)
    input = torch.tensor(rng.uniform(-1, 13, (9, 9, 20, 52)), dtype=torch.float16)
    weight = torch.tensor(rng.uniform(-2, -1, (9, 11, 12, 84)), dtype=torch.float16)
    stride = 1
    padding = 12

    out_cpu = torch.nn.functional.conv_transpose2d(input, weight, stride=stride, padding=padding)
    out_gpu = torch.nn.functional.conv_transpose2d(input.cuda(), weight.cuda(), stride=stride, padding=padding)
    
    assert not torch.equal(out_cpu, out_gpu.cpu())