import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71085
    input_tensor = torch.rand([10, 3, 5, 0], dtype=torch.float32)
    batch1 = torch.rand([10, 3, 4], dtype=torch.float32)
    batch2 = torch.rand([10, 4, 5], dtype=torch.float32)
    
    input_clone = input_tensor.clone()
    
    result = input_clone.baddbmm_(batch1, batch2)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == input_tensor.shape