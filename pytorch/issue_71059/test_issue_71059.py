import pytest
import torch

def test_f():
    input = torch.rand([])
    dim = 0
    index = torch.tensor([]) # or torch.tensor([0])
    src = torch.rand([])
    res = torch.scatter(input, dim, index, src)
    # random value like tensor(6.7333e+22)
    print(res)
    assert res is not None