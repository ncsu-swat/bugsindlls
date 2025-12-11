import pytest
import torch

def test_f1():
    input = torch.rand(torch.Size([2, 2]))
    dim = -1
    output = torch.nn.functional.softmax(input, dim=dim)
    print("succeed")
    assert output is not None
    
    input = input.to_sparse()
    with pytest.raises(RuntimeError) as e_info:
        res2 = torch.sparse.softmax(input, dim=dim).to_dense()
        # RuntimeError: : dim must be non-negative and less than input dimensions
    print(f"{e_info.type.__name__}: {e_info.value}")
    
def test_f2():
    input = torch.rand(torch.Size([2, 2]))
    dim = -1
    output = torch.nn.functional.log_softmax(input, dim=dim)
    print("succeed")
    assert output is not None
    
    input = input.to_sparse()
    with pytest.raises(RuntimeError) as e_info:
        res2 = torch.sparse.log_softmax(input, dim=dim).to_dense()
        # RuntimeError: : dim must be non-negative and less than input dimensions
    print(f"{e_info.type.__name__}: {e_info.value}")