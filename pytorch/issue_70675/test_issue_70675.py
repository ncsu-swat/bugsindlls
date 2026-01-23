import pytest
import torch

def test_f():
    a = torch.rand([0, 4])
    dim = 0
    indices = torch.tensor([0, 1])

    result = torch.index_select(a, dim, indices)
    print(result)
    assert result is not None