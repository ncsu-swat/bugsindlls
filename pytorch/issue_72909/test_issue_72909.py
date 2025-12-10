import pytest
import torch
import torch.nn as nn

def test_f():
    # 72909
    A = torch.rand([8, 5], dtype=torch.float32)
    
    with pytest.raises(AssertionError) as excinfo:
        torch.svd_lowrank(A)
        
    assert "AssertionError" in str(excinfo.type)