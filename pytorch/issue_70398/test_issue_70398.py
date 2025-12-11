import pytest
import torch

def test_f1():
    
    input = torch.rand([3])
    shape = [-2, 3]
    res = torch.broadcast_to(input,shape,)
    print(res.shape)
    # torch.Size([-2, 3])
    print(torch.sum(res))
    # tensor(0.)
    with pytest.raises(RuntimeError) as e_info:
        torch.all(res)
        # RuntimeError: Trying to create tensor with negative dimension -2: [-2, 3]
    print(f"{e_info.type.__name__}: {e_info.value}")
    
def test_f2():
    
    input = torch.rand([3])
    shape = [-2, 3]
    res = input.expand(shape)
    print(res.shape)
    print(torch.sum(res))
    with pytest.raises(RuntimeError) as e_info:
        torch.all(res)
        # RuntimeError: Trying to create tensor with negative dimension -2: [-2, 3]
    print(f"{e_info.type.__name__}: {e_info.value}")