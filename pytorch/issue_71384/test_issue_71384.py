import pytest
import torch
import torch.nn.functional as F

def test_f1():

    A = torch.rand([0, 4, 4, 3, 0], dtype=torch.float64)
    B = torch.rand([0, 3, 3], dtype=torch.float64)
    tensorsolve_res=torch.linalg.tensorsolve(A, B)
    print(tensorsolve_res)
    assert tensorsolve_res is not None
    # tensor([], size=(3, 0), dtype=torch.float64)
    x = torch.linalg.tensorinv(A)
    with pytest.raises(RuntimeError) as e_info:
        torch.tensordot(x, B)
    print(f"{e_info.type.__name__}: {e_info.value}")
    
def test_f2():

    A= torch.rand([6, 4, 4, 3, 2], dtype=torch.float64)
    B= torch.rand([6, 4, 2], dtype=torch.float64)
    x = torch.linalg.tensorinv(A)
    tensordot_res=torch.tensordot(x, B)
    print("success")
    assert tensordot_res is not None
    
    with pytest.raises(RuntimeError) as e_info:
        print(torch.linalg.tensorsolve(A, B))
    print(f"{e_info.type.__name__}: {e_info.value}")