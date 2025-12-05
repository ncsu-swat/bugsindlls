import torch
import numpy as np
import pytest

def test_f():
    rng = np.random.default_rng(716)

    input_tensor = torch.tensor(rng.uniform(-9223320268019550000., -9223320268019550000., size=(1, 1, 1, 1, 3, 3)), dtype=torch.complex128)

    output_cpu = torch.det(input_tensor)
    output_gpu = torch.det(input_tensor.to("cuda"))

    print(output_cpu[0,0,0,0])
    # tensor(0.+0.j, dtype=torch.complex128)
    print(output_gpu[0,0,0,0])
    # tensor(-1.0737e+09+0.j, device='cuda:0', dtype=torch.complex128)
    
    assert not torch.allclose(output_cpu.cpu(), output_gpu.cpu())