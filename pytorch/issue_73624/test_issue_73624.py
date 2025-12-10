import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    # 73624
    kernel_size = [2, 2]
    output_size = [0, 1]
    input_tensor = torch.rand([16, 50, 44, 31], dtype=torch.float32)
    
    with pytest.raises(RuntimeError) as excinfo:
        nn.FractionalMaxPool2d(kernel_size, output_size=output_size)(input_tensor)
        
    assert "Expected output_size to be non-negative" in str(excinfo.value) or "invalid output size" in str(excinfo.value)