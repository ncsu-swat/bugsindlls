import pytest
import torch
import torch.nn as nn

def test_f():
    # 72821
    input_tensor = torch.rand([1, 0, 2, 2])
    num_features = 5
    
    batchnorm = nn.BatchNorm2d(num_features)

    result = batchnorm(input_tensor)

    assert isinstance(result, torch.Tensor)
    assert result.shape == input_tensor.shape