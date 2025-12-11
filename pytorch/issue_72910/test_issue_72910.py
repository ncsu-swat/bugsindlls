import pytest
import torch
import torch.nn as nn

def test_f():
    results = dict()
    input = torch.rand([2, 1])
    with pytest.raises(RuntimeError) as e_info:
        torch.fft.irfftn(input)
        torch.fft.irfft2(input)
    print(f"{e_info.type.__name__}: {e_info.value}")