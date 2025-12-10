import pytest
import torch

def test_f():
    input_dense = torch.rand(torch.Size([2, 2]))
    dim = -1

    input_sparse = input_dense.to_sparse()

    with pytest.raises(RuntimeError) as excinfo:
        torch.sparse.softmax(input_sparse, dim=dim).to_dense()

    assert "dim must be non-negative and less than input dimensions" in str(excinfo.value)