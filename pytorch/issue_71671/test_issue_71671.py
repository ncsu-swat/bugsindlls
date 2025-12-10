import pytest
import torch

def test_f():
    # 71671
    input_tensor = torch.rand([1], dtype=torch.float64)
    other = torch.rand([1], dtype=torch.complex32)
    
    with pytest.raises(RuntimeError) as excinfo:
        torch.add(input_tensor, other)
        
    assert "INTERNAL ASSERT FAILED" in str(excinfo.value)