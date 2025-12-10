import pytest
import torch
import sys

def test_f():
    input_dense = torch.rand([1, 1])
    mat1_dense = torch.rand([2, 3])
    mat2 = torch.rand([3, 3])

    input_sparse = input_dense.to_sparse()
    mat1_sparse = mat1_dense.to_sparse()

    with pytest.raises(RuntimeError) as excinfo:
        torch.sspaddmm(input_sparse, mat1_sparse, mat2)

    assert "Expected dim 0 size 2, got 1" in str(excinfo.value)