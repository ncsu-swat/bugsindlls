import pytest
import torch
import numpy as np

def test_f():

    torch.random.manual_seed(420)
    input = torch.randn(3, 3, dtype=torch.float32)
    print("Intermediate: ", torch.log(input * 2 - 1)) # Contains nan
    cpu_output = torch.matrix_exp(torch.log(input * 2 - 1))
    print("cpu output: ", cpu_output)
    input = input.cuda()
    gpu_output = torch.matrix_exp(torch.log(input * 2 - 1))
    print("gpu output: ", gpu_output) 
    assert not torch.allclose(cpu_output.cuda(), gpu_output)