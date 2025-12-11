import pytest
import torch

def test_f():
    condition = torch.randint(0, 2, [2, 2], dtype=torch.bool)
    x = torch.rand([2, 2], dtype=torch.float64)
    y = 0.0
    res=torch.where(condition, x, y)
    print(res)
    assert res is not None
    # tensor([[0.0000, 0.6290],
    #        [0.0000, 0.0000]], dtype=torch.float64)
    
    with pytest.raises(TypeError) as e_info:
        print( x.where(condition, y) )
    print(f"{e_info.type.__name__}: {e_info.value}")