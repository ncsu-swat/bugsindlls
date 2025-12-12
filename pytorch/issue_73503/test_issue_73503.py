import pytest
import torch

def test_f():
    input = torch.rand([0, 1])
    
    with pytest.raises(ZeroDivisionError) as e_info:
        torch.nn.init.orthogonal_(input)
    print(f"{e_info.type.__name__}: {e_info.value}")