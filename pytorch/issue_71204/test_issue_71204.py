import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71204
    a = torch.tensor([[0, 1], [2, 3]])
    diagonal_offset = 3
    
    with pytest.raises(RuntimeError) as excinfo:
        torch.diag(a, diagonal_offset)
        
    assert "alloc_cpu() seems to have been called with negative number" in str(excinfo.value)