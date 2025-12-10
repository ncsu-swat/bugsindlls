import pytest
import torch

def test_f():
    condition = torch.randint(0, 2, [2, 2], dtype=torch.bool)
    x = torch.rand([2, 2], dtype=torch.float64)
    y = 0.0

    with pytest.raises(TypeError) as excinfo:
        x.where(condition, y)
    
    assert "must be Tensor, not float" in str(excinfo.value)