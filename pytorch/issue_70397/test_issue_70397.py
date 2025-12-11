import pytest
import torch

def test_f():
    size = [2, 3]
    stride = [-1, 2]
    
    res = torch.empty_strided(size, stride)
    print(torch.sum(res))
    print(res.shape)
    with pytest.raises(RuntimeError) as e_info:
        print(res)
    print(f"{e_info.type.__name__}: {e_info.value}")