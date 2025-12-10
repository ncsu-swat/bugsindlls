import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    # 73503
    input_tensor = torch.rand([0, 1])
    
    with pytest.raises(ZeroDivisionError) as excinfo:
        nn.init.orthogonal_(input_tensor)
        
    assert "integer division or modulo by zero" in str(excinfo.value)