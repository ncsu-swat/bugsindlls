import pytest
import torch
import torch.nn as nn

def test_f():
    # 72824
    a = torch.rand([3], dtype=torch.float32)
    b = torch.rand([3], dtype=torch.float64)
    
    bce_loss = nn.BCELoss()

    with pytest.raises(RuntimeError) as excinfo:
        bce_loss(a, b)
        
    assert "Found dtype Double but expected Float" in str(excinfo.value)