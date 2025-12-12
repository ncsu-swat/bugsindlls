import pytest
import torch

def test_f():

    input = torch.rand([1, 1], dtype=torch.complex32)
    with pytest.raises(RuntimeError) as e_info:
        input.storage()
    print(f"{e_info.type.__name__}: {e_info.value}")