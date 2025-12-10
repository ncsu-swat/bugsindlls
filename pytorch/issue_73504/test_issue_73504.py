import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    # 73504
    input_tensor = torch.randint(0, 4, [], dtype=torch.uint8)
    value = 2110
    
    with pytest.raises(RuntimeError) as excinfo:
        torch.ge(input_tensor, value)
        
    assert "value cannot be converted to type uint8_t without overflow" in str(excinfo.value)