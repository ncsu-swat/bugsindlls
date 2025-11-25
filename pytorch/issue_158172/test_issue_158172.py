import torch
import pytest

def test_f():
    input_tensor = torch.tensor([-2000000], dtype=torch.int64)
    other = torch.tensor([-1])
    output_cpu = torch.tensor([], dtype=torch.int16)
    output_gpu = torch.tensor([], dtype=torch.int16).cuda()

    cpu_res=torch.fmin(input_tensor, other, out=output_cpu)
    gpu_res=torch.fmin(input_tensor.cuda(), other.cuda(), out=output_gpu)

    print(cpu_res)   # 31616
    print(gpu_res)   # -1
        
    assert not torch.equal(cpu_res, gpu_res.cpu()), "CPU and GPU results match"