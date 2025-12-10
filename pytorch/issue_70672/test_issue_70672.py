import pytest
import torch

def test_f():
    a = torch.rand([3, 3])
    
    b = torch.as_strided(a, [1, -1], [1, 1])
    
    with pytest.raises(RuntimeError) as excinfo:
        print(b)
    
    assert "Trying to create tensor with negative dimension" in str(excinfo.value)