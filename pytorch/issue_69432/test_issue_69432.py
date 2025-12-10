import pytest
import torch

def test_f():
    tensor = torch.rand(torch.Size([]))
    
    source_dim = 0
    destination_dim = 0

    with pytest.raises(RuntimeError) as excinfo:
        torch.movedim(tensor, source_dim, destination_dim)

    assert "INTERNAL ASSERT FAILED" in str(excinfo.value)