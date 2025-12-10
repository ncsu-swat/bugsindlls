import pytest
import torch

def test_f():
    # 71059
    input_tensor = torch.rand([])
    dim = 0
    index = torch.tensor([])
    src = torch.rand([])

    result = torch.scatter(input_tensor, dim, index, src)

    bug_reproduced = result.item() != 0.0
    
    assert bug_reproduced