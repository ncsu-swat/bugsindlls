import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71385
    A = torch.rand([2, 3, 3], dtype=torch.float64)
    B = torch.rand([2, 3], dtype=torch.float64)
    
    res=torch.linalg.solve(A, B)
    print(res)
    assert res is not None
    
    with pytest.raises(RuntimeError) as e_info:
        torch.linalg.inv(A) @ B
    print(f"{e_info.type.__name__}: {e_info.value}")