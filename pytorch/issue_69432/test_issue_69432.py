import pytest
import torch

def test_f():

    tensor = torch.rand(torch.Size([]))
    with pytest.raises(RuntimeError) as e_info:
        res1 = torch.movedim(tensor, 0, 0)
    print(f"{e_info.type.__name__}: {e_info.value}")