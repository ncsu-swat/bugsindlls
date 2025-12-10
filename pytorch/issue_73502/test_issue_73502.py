import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    # 73502
    input_tensor = torch.rand([1, 1], dtype=torch.complex32)
    
    with pytest.raises(RuntimeError) as excinfo:
        input_tensor.storage()
        
    assert "unsupported Storage type" in str(excinfo.value)