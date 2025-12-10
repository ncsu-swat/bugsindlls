import pytest
import torch

def test_f():
    # 71636
    input_tensor = torch.randint(-2, 2, [0], dtype=torch.int32)
    
    result = torch.median(input_tensor)
    
    assert result.item() == -2147483648