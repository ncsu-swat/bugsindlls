import pytest
import torch

def test_f():
    # 71472
    input_tensor = torch.rand([6, 5, 5], dtype=torch.complex128)

    with pytest.raises(RuntimeError) as excinfo:
        torch.nanmean(input_tensor)
        
    assert "expected input to have floating point dtype but got ComplexDouble" in str(excinfo.value)