import pytest
import torch

def test_f1():
    zero_tensor = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    output = torch.mm(sparse_zero_tensor, mat1) 
    print("Success")
    assert output is not None
    with pytest.raises(RuntimeError) as e_info:
        sparse_zero_tensor = sparse_zero_tensor + torch.tensor([[0, 1], [0, 0]], dtype=torch.int64).to_sparse()
        torch.mm(sparse_zero_tensor, mat1)
    print(f"{e_info.type.__name__}: {e_info.value}")
    
def test_f2():
    zero_tensor = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    output = torch.sparse.mm(sparse_zero_tensor, mat1) 
    print("Success")
    assert output is not None
    with pytest.raises(RuntimeError) as e_info:
        torch.sparse.mm(zero_tensor, mat1)
    print(f"{e_info.type.__name__}: {e_info.value}")

def test_f3():
    zero_tensor = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    output = torch.matmul(sparse_zero_tensor, mat1) 
    print("Success")
    assert output is not None
    with pytest.raises(RuntimeError) as e_info:
        torch.matmul(zero_tensor, mat1)
    print(f"{e_info.type.__name__}: {e_info.value}")

def test_f4():
    zero_tensor = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.tensor([1, 2], dtype=torch.int32)
    output = torch.mv(sparse_zero_tensor, mat1) 
    print("Success")
    assert output is not None
    with pytest.raises(RuntimeError) as e_info:
        torch.mv(zero_tensor, mat1)
    print(f"{e_info.type.__name__}: {e_info.value}")
    
def test_f5():
    zero_tensor = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    output = torch.addmm(mat1, sparse_zero_tensor, mat1) 
    print("Success")
    assert output is not None
    with pytest.raises(RuntimeError) as e_info:
        torch.addmm(mat1, zero_tensor, mat1)
    print(f"{e_info.type.__name__}: {e_info.value}")

def test_f6():
    zero_tensor = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    output = torch.sparse.addmm(mat1, sparse_zero_tensor, mat1) 
    print("Success")
    assert output is not None
    with pytest.raises(RuntimeError) as e_info:
        torch.sparse.addmm(mat1, zero_tensor, mat1)
    print(f"{e_info.type.__name__}: {e_info.value}")

def test_f7():
    zero_tensor = torch.randint(0, 1, [1,1,1], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.randint(0, 10, [1, 1, 1], dtype=torch.int32)
    mat2 = torch.randint(0, 10, [1, 1, 1], dtype=torch.int32)
    output = torch.bmm(sparse_zero_tensor, mat1)
    print("Success")
    assert output is not None
    with pytest.raises(RuntimeError) as e_info:
        torch.bmm(zero_tensor, mat1)
    print(f"{e_info.type.__name__}: {e_info.value}")

def test_f8():
    zero_tensor = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    sparse_zero_tensor = zero_tensor.to_sparse()
    mat1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    with pytest.raises(RuntimeError) as e_info:
        torch.smm(sparse_zero_tensor, mat1)
        torch.sspaddmm(sparse_zero_tensor, sparse_zero_tensor, mat1)
    print(f"{e_info.type.__name__}: {e_info.value}")