import pytest
import torch

def test_f():
    arg_1 = torch.rand([5, 5], dtype=torch.float64)
    arg_2 = torch.rand([5, 5], dtype=torch.float64)
    arg_3 = torch.rand([1, 5], dtype=torch.complex128)
    
    with pytest.raises(RuntimeError) as e_info:
        res = torch.addcmul(arg_1,arg_2,arg_3)
    print(f"{e_info.type.__name__}: {e_info.value}")