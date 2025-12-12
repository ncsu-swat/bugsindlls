import pytest
import torch

def test_f():
    input = torch.rand([5, 5, 0])
    res=torch.median(input)
    print(res)
    assert torch.isnan(res)
    with pytest.raises(RuntimeError) as e_info:
        torch.max(input)
        torch.min(input)
    print(f"{e_info.type.__name__}: {e_info.value}")