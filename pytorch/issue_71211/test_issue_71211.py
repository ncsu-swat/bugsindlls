import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71211
    input_tensor = torch.rand([2, 2])
    
    result = F.relu(input_tensor, inplace="aaaa")
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == input_tensor.shape