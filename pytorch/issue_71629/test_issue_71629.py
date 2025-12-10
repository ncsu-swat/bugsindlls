import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    # 71629
    input_tensor = torch.rand([5, 5, 0])

    with pytest.raises(RuntimeError) as excinfo:
        torch.max(input_tensor)
        
    assert "Expected reduction dim to be specified for input.numel() == 0" in str(excinfo.value)