import pytest
import torch

def test_f():
    arg_1 = torch.rand([5, 5], dtype=torch.float64)
    arg_2 = torch.rand([5, 5], dtype=torch.float64)
    arg_3 = torch.rand([1, 5], dtype=torch.complex128)
    
    with pytest.raises(RuntimeError) as excinfo:
        torch.addcmul(arg_1, arg_2, arg_3)
    
    assert "INTERNAL ASSERT FAILED" in str(excinfo.value)