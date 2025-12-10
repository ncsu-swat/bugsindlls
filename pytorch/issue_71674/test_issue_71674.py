import pytest
import torch

def test_f():
    # 71674
    d = torch.complex64
    
    with pytest.raises(RuntimeError) as excinfo:
        torch.set_default_dtype(d)
        
    assert "only floating-point types are supported" in str(excinfo.value) or "invalid dtype object" in str(excinfo.value)