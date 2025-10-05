import torch
import pytest
import numpy as np

def test_f():
    a = torch.tensor(float('nan'))

    cpu_output = a.clone().cpu().type(torch.int32)
    gpu_output = a.clone().cuda().type(torch.int32)
    print(cpu_output) # tensor(-2147483648, dtype=torch.int32)
    print(gpu_output) # tensor(0, device='cuda:0', dtype=torch.int32)

    assert not torch.equal(cpu_output, gpu_output.cpu())