import pytest
import torch

def test_f():

    results = dict()
    input_tensor = torch.rand([], dtype=torch.float64)

    input = input_tensor.clone()
    dim = 0
    sum_res=torch.sum(input, dim=dim)
    print(sum_res)
    assert sum_res is not None
    # tensor(0.1512, dtype=torch.float64)
    
    with pytest.raises(RuntimeError) as e_info:
        torch.sparse.sum(input_tensor.clone().to_sparse(),dim=dim,)
        # RuntimeError: Trying to create tensor with negative dimension -1: [-1, 1]
    print(f"{e_info.type.__name__}: {e_info.value}")