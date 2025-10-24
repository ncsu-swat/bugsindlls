import torch
import pytest
import torch.nn as nn

def test_f():

    def forward(x):
        return torch.unfold_copy(dimension=1, input=x,size=0,step=7)
    
    x = torch.rand([1,0], dtype=torch.float32)# generate arg
    forward(x)# on eagermode
    print("build succeeded")
    with pytest.raises(ZeroDivisionError) as e_info:
        res=torch.compile(forward, mode='max-autotune',fullgraph=True)(x)
        print("compile mode result", res)
    print(f'{e_info.type.__name__}: {e_info.value}')