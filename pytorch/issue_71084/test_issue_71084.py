import pytest
import torch
import torch.nn.functional as F

def test_f():
    results = dict()
    input = torch.rand([0, 1, 2], dtype=torch.float64)
    
    with pytest.raises(RuntimeError) as e_info:
        torch.cholesky_inverse(input)
        torch.det(input)   
    print(f"{e_info.type.__name__}: {e_info.value}")
    res1=torch.cholesky(input)
    print(res1)
    res2=torch.inverse(input)
    print(res2)