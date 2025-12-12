import pytest
import torch

def test_f():
    results = dict()
    input = torch.randint(0,4,[], dtype=torch.uint8)
    value = 2110
    res=torch.div(input, value, rounding_mode="trunc")
    print(res)
    # tensor(0, dtype=torch.uint8)
    with pytest.raises(RuntimeError) as e_info:
        torch.ge(input, value)
    print(f"{e_info.type.__name__}: {e_info.value}")