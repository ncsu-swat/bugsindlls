import pytest
import torch

def test_f():

    input = torch.rand([1, 1])
    mat1 = torch.rand([2, 3])
    mat2 = torch.rand([3, 3])
    res1 = torch.addmm(input, mat1, mat2)
    print("addmm pass")
    assert res1 is not None
    input = input.to_sparse()
    mat1 = mat1.to_sparse()
    with pytest.raises(RuntimeError) as e_info:
        res2 = torch.sspaddmm(input, mat1, mat2)
    print(f"{e_info.type.__name__}: {e_info.value}")