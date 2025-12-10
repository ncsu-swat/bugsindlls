import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71080
    filters = torch.randn(8, 4, 3, 3)
    inputs = torch.randn(1, 4, 5, 5)
    
    with pytest.raises(RuntimeError) as excinfo:
        F.conv2d(inputs, filters, padding=1, groups=0)
    
    assert "groups must be greater than zero" in str(excinfo.value) or "expected groups to be in range" in str(excinfo.value)