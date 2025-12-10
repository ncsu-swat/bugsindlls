import pytest
import torch
import torch.nn as nn

def test_f():
    # Reproduces Bug #70232: AdaptiveAvgPool{2|3}d creates tensor with negative dimension
    output_size = [-36, 0]
    avg_pool = nn.AdaptiveAvgPool2d(output_size)
    
    tensor = torch.rand([128, 2048, 4, 4], dtype=torch.float32)
    
    result = avg_pool(tensor)
    
    bug_reproduced = any(dim < 0 for dim in result.shape)
    
    assert bug_reproduced, (
        f"BUG NOT REPRODUCED: The output shape {result.shape} does not contain a negative dimension."
    )