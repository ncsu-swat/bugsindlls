import torch
import pytest

def test_f():
    output_gpu_scalar = torch.logcumsumexp(torch.tensor(22, dtype=torch.int64).cuda(), 0) # no runtime error
    print("GPU scalar output:", output_gpu_scalar)

    with pytest.raises(RuntimeError) as e_info:
        output_gpu = torch.logcumsumexp(torch.tensor([22], dtype=torch.int64).cuda(), 0) # error
        output_cpu_scalar = torch.logcumsumexp(torch.tensor(22, dtype=torch.int64), 0) # error
    print(f'{e_info.type.__name__}: {e_info.value}')