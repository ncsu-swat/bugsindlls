import pytest
import torch

def test_f():
    size = [2, 3]
    stride = [-1, 2]
    
    res = torch.empty_strided(size, stride)
    
    with pytest.raises(RuntimeError) as excinfo:
        print(res)
    
    assert "out of bounds for storage of size" in str(excinfo.value)