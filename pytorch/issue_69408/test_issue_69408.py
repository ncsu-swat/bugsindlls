import pytest
import torch

def test_f():
    tensor_0 = torch.rand(torch.Size([]))
    tensor_1 = torch.rand(torch.Size([3]))
    tensors = [tensor_0, tensor_1]

    res1 = torch.hstack(tensors)
    print("Res1 succeeded")
    assert res1 is not None
    
    with pytest.raises(RuntimeError) as e_info:
        res2 = torch.cat(tensors, dim=0)
    print(f"{e_info.type.__name__}: {e_info.value}")