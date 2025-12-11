import pytest
import torch

def test_f():
    mat = torch.randint(0, 2, torch.Size([2, 3]), dtype=torch.int64)
    mat1 = torch.randint(0, 1, torch.Size([2, 3]), dtype=torch.int64) # empty tensor
    mat2 = torch.randint(0, 2, torch.Size([3, 3]), dtype=torch.int32)

    mat1_sparse = mat1.to_sparse()
    print(mat1_sparse.dtype)
    # torch.int64
    sparse_output = torch.sparse.addmm(mat, mat1_sparse, mat2)
    print(sparse_output)
    assert sparse_output is not None
    
    with pytest.raises(RuntimeError) as e_info:
        normal_output = torch.addmm(mat, mat1, mat2)
        # RuntimeError: expected scalar type Long but found Int
    print(f"{e_info.type.__name__}: {e_info.value}")