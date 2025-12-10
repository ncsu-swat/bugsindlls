import pytest
import torch

def test_f():
    # 70675
    a = torch.rand([0, 4])
    dim = 0
    indices = torch.tensor([0, 1])

    result = torch.index_select(a, dim, indices)

    bug_reproduced = (result.numel() > 0) and (torch.count_nonzero(result) > 0)
    
    assert bug_reproduced