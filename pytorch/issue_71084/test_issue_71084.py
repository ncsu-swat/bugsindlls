import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71084
    input_tensor = torch.rand([0, 1, 2], dtype=torch.float64)
    
    result = torch.inverse(input_tensor)
    
    assert isinstance(result, torch.Tensor) and result.numel() == 0 and result.dim() == 3