import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71384
    A = torch.rand([0, 4, 4, 3, 0], dtype=torch.float64)
    B = torch.rand([0, 3, 3], dtype=torch.float64)
    
    X = torch.linalg.tensorinv(A)
    
    with pytest.raises(RuntimeError) as excinfo:
        torch.tensordot(X, B)
        
    assert "contracted dimensions need to match" in str(excinfo.value)
