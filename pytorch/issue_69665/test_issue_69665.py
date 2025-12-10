import pytest
import torch

def test_f():
    # Reproduces Bug #69665: torch.mm does not check dtype for zero sparse tensor
    zero_tensor = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    output = torch.mm(sparse_zero_tensor, mat1)
    
    assert isinstance(output, torch.Tensor)
    
    assert output.shape == torch.Size([2, 2])