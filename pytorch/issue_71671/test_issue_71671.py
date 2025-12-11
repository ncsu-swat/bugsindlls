import pytest
import torch

def test_f():

    results = dict()
    input = torch.rand([1], dtype=torch.float64)
    other = torch.rand([1], dtype=torch.complex32)
    with pytest.raises(RuntimeError) as e_info:
        results["res"] = torch.add(input, other)
        print(results)
    print(f"{e_info.type.__name__}: {e_info.value}")