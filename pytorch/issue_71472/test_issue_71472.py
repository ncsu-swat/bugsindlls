import pytest
import torch

def test_f1():

    input= torch.rand([6, 5, 5], dtype=torch.complex128)
    mean_res=torch.mean(input)
    print(mean_res)
    assert mean_res is not None
    # tensor(0.5126+0.5373j, dtype=torch.complex128)
    with pytest.raises(RuntimeError) as e_info:
        torch.nanmean(input)
        # RuntimeError: nanmean(): expected input to have floating point dtype but got ComplexDouble
    print(f"{e_info.type.__name__}: {e_info.value}")

def test_f2():

    input = torch.rand([2, 2], dtype=torch.complex64)
    input[0] = torch.nan
    mean_res=torch.mean(input)
    print(mean_res)
    assert mean_res is not None
    with pytest.raises(RuntimeError) as e_info:
        torch.nanmean(input)
        # RuntimeError: nanmean(): expected input to have floating point dtype but got ComplexDouble
    print(f"{e_info.type.__name__}: {e_info.value}")