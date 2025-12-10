import pytest
import torch

def test_f():
    # 71477
    input_tensor = torch.rand([], dtype=torch.float64)
    dim = 100

    result_values, result_indices = torch.cummin(input_tensor, dim)

    bug_reproduced = (result_values.dim() == 0) and (result_indices.dim() == 0)
    
    assert bug_reproduced