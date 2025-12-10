import pytest
import torch

def test_f():
    # 72793
    input_tensor = torch.rand([], dtype=torch.float64)
    dim = 0
    
    input_sparse = input_tensor.to_sparse()

    with pytest.raises(RuntimeError) as excinfo:
        torch.sparse.sum(input_sparse, dim=dim)
        
    assert "Trying to create tensor with negative dimension" in str(excinfo.value)