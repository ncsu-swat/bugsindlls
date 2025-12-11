import pytest
import torch

def test_f():
    a = torch.rand([3, 3])   
    b = torch.as_strided(a, [1, -1], [1, 1])
    print(b.shape)
    assert True
    with pytest.raises(RuntimeError) as e_info:
        print(b)
    print(f"{e_info.type.__name__}: {e_info.value}")