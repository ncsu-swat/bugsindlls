import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71385
    A = torch.rand([2, 3, 3], dtype=torch.float64)
    B = torch.rand([2, 3], dtype=torch.float64)
    
    A_inv = torch.linalg.inv(A)
    
    with pytest.raises(RuntimeError) as excinfo:
        A_inv @ B
        
    assert "mat1 and mat2 shapes cannot be multiplied" in str(excinfo.value)
