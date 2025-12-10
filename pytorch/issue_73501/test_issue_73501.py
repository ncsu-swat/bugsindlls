import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

def test_f():
    # 73501
    input_tensor = torch.rand([], dtype=torch.float64)
    dim = 0
    index = torch.tensor(-1, dtype=torch.int64)
    value = torch.rand([], dtype=torch.float64)

    with pytest.raises(IndexError) as excinfo:
        input_tensor.clone().index_copy(dim, index, value)

    assert "is out of bounds for dimension 0 with size 1" in str(excinfo.value)