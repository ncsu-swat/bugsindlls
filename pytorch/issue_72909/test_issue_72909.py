import pytest
import torch
import torch.nn as nn

def test_f():
    
    A = torch.rand([8, 5], dtype=torch.float32)
    res=torch.pca_lowrank(A)
    print("Success")
    assert res is not None
    with pytest.raises(AssertionError) as e_info:
        torch.svd_lowrank(A)
        print("Success")
    print(f"{e_info.type.__name__}: {e_info.value}")