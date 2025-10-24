import torch
import pytest
import numpy as np
import torch.nn as nn
import traceback

def test_f():
    def fn(x, y, z):
        return torch.mm(x, y, out=z)

    inputs = [torch.rand((4, 4)) for _ in range(3)]

    with pytest.raises(RuntimeError) as e_info:
        fn_opt = torch.compile(fn, dynamic=True)
        fn_opt(*inputs)
    print(f'{e_info.type.__name__}: {e_info.value}')