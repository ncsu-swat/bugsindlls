import torch
import pytest
import numpy as np

def test_f():
    input = torch.FloatTensor([1, 3])
    cpu_output = torch.unique(input, sorted=False)
    print(cpu_output)

    gpu_output = torch.unique(input.cuda(), sorted=False)
    print(gpu_output)

    assert not torch.equal(cpu_output, gpu_output.cpu())
    
