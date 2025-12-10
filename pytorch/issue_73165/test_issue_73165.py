import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    # 73165
    input_1 = torch.rand([5, 0], dtype=torch.float32)
    input_2 = torch.rand([5, 0], dtype=torch.float32)
    
    loss_fn = nn.CrossEntropyLoss()
    
    with pytest.raises(RuntimeError) as excinfo:
        loss_fn(input_1, input_2)
        
    assert "Expected target size" in str(excinfo.value) or "Expected input to be non-empty" in str(excinfo.value) or "division by zero" in str(excinfo.value)