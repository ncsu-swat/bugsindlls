import torch
import numpy as np
import pytest

def test_f():
    rng = np.random.default_rng(432)

    input_tensor = torch.tensor(rng.uniform(-20260., -19740., size=(18, 1, 2, 2, 21, 2)), dtype=torch.float16)
    dim = 2

    output_cpu = torch.nn.LogSoftmax(dim=dim)(input_tensor)
    output_gpu = torch.nn.LogSoftmax(dim=dim).cuda()(input_tensor.cuda())

    print(output_cpu[0, 0, 0, 0, 5, 0])  # tensor(-inf, dtype=torch.float16)
    print(output_gpu[0, 0, 0, 0, 5, 0])  # tensor(0., device='cuda:0', dtype=torch.float16)
    
    assert not torch.allclose(output_cpu, output_gpu.cpu()), "CPU and GPU outputs are close!"