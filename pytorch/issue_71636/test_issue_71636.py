import pytest
import torch

def test_f():

    input = torch.randint(-2,2,[0], dtype=torch.int32)
    median_res=torch.median(input)
    print(median_res)
    assert median_res is not None
    # tensor(-2147483648, dtype=torch.int32)
    with pytest.raises(RuntimeError) as e_info:
        torch.min(input)
        # RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
    print(f"{e_info.type.__name__}: {e_info.value}")