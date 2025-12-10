import pytest
import torch

def test_f():
    input_tensor = torch.rand([3])
    shape = [-2, 3]
    
    res = torch.broadcast_to(input_tensor, shape)
    
    with pytest.raises(RuntimeError) as excinfo:
        torch.all(res)
    
    assert "Trying to create tensor with negative dimension" in str(excinfo.value)