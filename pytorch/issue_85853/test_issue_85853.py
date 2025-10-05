import torch
import pytest

def test_f():
    torch.manual_seed(420)
    torch.cuda.manual_seed_all(420)

    x = torch.randn(10, 10).log()  # contains NaN
    cpu_histc = torch.histc(x, bins=10, min=0, max=1)
    print("CPU:", cpu_histc)

    x = x.cuda()
    gpu_histc = torch.histc(x, bins=10, min=0, max=1)
    print("GPU:", gpu_histc)
    print("NaN count:", torch.isnan(x.view(-1)).sum().item())

    # Check mismatch in histc results
    assert not torch.equal(cpu_histc, gpu_histc.cpu())
