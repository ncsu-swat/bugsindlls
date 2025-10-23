import torch
import torch.nn as nn
import pytest

def test_f():

    m = nn.AdaptiveMaxPool1d(1)
    inputs = torch.rand([0,1,0])
    with pytest.raises(RuntimeError) as e_info: 
        m(inputs)
    print(f'{e_info.type.__name__}: {e_info.value}')