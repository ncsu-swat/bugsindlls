import torch
import torch.nn as nn
import pytest

def test_f():

    input_size = [1,2]
    m = nn.CrossMapLRN2d(1,0.1,0.1,1)
    
    with pytest.raises(AssertionError) as e_info: 
        m(torch.rand(input_size))
    print(f'{e_info.type.__name__}: {e_info.value}')
