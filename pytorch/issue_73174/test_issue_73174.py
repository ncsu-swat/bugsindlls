import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    # 73174
    normalized_shape = [1024]
    layer = nn.LayerNorm(normalized_shape)
    
    input_tensor = torch.randint(0, 8, [0, 1, 1024], dtype=torch.long)
    
    with pytest.raises(RuntimeError) as excinfo:
        layer(input_tensor)
        
    assert "INTERNAL ASSERT FAILED" in str(excinfo.value)