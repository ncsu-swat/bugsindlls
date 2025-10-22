import torch.nn as nn
import torch
import pytest

def test_f1():

    m = nn.ConstantPad3d(padding=1, value=1)

    with pytest.raises(RuntimeError) as e_info:  
        m(torch.rand([1,2]))
    print(f'{e_info.type.__name__}: {e_info.value}')

def test_f2():

    l = nn.ConstantPad2d(padding=1, value=1)

    with pytest.raises(RuntimeError) as e_info: 
        l(torch.rand([2]))
    print(f'{e_info.type.__name__}: {e_info.value}') 
    