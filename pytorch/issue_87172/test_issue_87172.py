import torch
import pytest
import numpy as np

def test_f():
    x = torch.ones(2)
    y = torch.sin(x)
    y = torch.arcsin(y)  # inverse operations, gives [1.0000, 1.0000]
    print(y)
    cpu_output = torch.arccos(y)
    print(cpu_output) # gives [0.0003, 0.0003]

    x1 = x.clone().cuda()
    y1 = torch.sin(x1)
    y1 = torch.arcsin(y1) # gives [1.0000, 1.0000]
    print(y1)
    gpu_output = torch.arccos(y1)
    print(gpu_output) # gives [nan, nan]

    assert not torch.equal(cpu_output, gpu_output.cpu())