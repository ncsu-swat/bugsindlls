import torch
import pytest

def test_f():
    xGpu = torch.tensor([-10.0, 10.0]).cuda()
    pGpu = torch.poisson(xGpu)
    print(pGpu)
 
    with pytest.raises(Exception) as e_info:
        x = torch.tensor([-10.0, 10.0])
        p = torch.poisson(x)
    print(f'{e_info.type.__name__}: {e_info.value}')
