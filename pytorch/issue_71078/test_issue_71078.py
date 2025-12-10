import pytest
import torch
import torch.nn as nn

def test_f():
    # 71078
    padding = [-1, -2, 1, 1]
    pad_module = nn.ConstantPad2d(padding, 0)
    input_tensor = torch.rand([1, 1, 3, 3], dtype=torch.float32)

    with pytest.raises(RuntimeError) as excinfo:
        pad_module(input_tensor)
    
    assert "resulted in a negative output size" in str(excinfo.value)