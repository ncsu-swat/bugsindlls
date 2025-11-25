import torch
import numpy as np
import pytest

def check(dtype):
    rng = np.random.default_rng(2186)
    input = torch.tensor(rng.uniform(-3.998, 5.0, (6, 7, 15, 23)), dtype=dtype)
    weight = torch.tensor(rng.uniform(-9990.0, -9990.0, (7, 3, 15, 18)), dtype=dtype)
    stride = 1
    padding = 2

    out_cpu = torch.nn.functional.conv_transpose2d(input, weight, stride=stride, padding=padding)
    out_gpu = torch.nn.functional.conv_transpose2d(input.cuda(), weight.cuda(), stride=stride, padding=padding)
    print(f"CPU output 1: {out_cpu[0,0,0,0]}")
    print(f"CPU output 2: {out_cpu[5, 2, 24, 0]}")
        
    print(f"GPU output 1: {out_gpu.cpu()[0,0,0,0]}")
    print(f"GPU output 2: {out_gpu.cpu()[5, 2, 24, 0]}")

    print(f"Float16 max value: {torch.finfo(torch.float16).max}") # 65504.0
    return out_cpu, out_gpu

def test_1(): 

    # float16: inconsistent value
    print("Float16 (inconsistent)")
    out_cpu, out_gpu=check(torch.float16)
    assert not torch.equal(out_cpu, out_gpu.cpu())
    
def test_2():

    # float32: consistent value
    print("Float32 (inconsistent)")
    out_cpu, out_gpu=check(torch.float32)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-4, rtol=1e-4)