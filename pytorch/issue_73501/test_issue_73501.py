import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    input_tensor = torch.rand([], dtype=torch.float64)
    dim = 0
    index = torch.tensor(-1, dtype=torch.int64)
    value = torch.rand([], dtype=torch.float64)
    res=input_tensor.clone().index_fill(dim, index, value)
    print(res)
    with pytest.raises(IndexError) as e_info:
        input_tensor.clone().index_copy(dim, index, value)
    print(f"{e_info.type.__name__}: {e_info.value}")