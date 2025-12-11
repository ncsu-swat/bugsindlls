import pytest
import torch

def test_f():
    
    shapes = [1, -12]
    res1 = torch.broadcast_shapes(*shapes)
    print(res1)
    assert res1 is not None
    with pytest.raises(RuntimeError) as e_info:
        res2 = torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape
        print(res2)
    print(f"{e_info.type.__name__}: {e_info.value}")