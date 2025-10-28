import torch
from torch import nn
import pytest
import numpy as np

def test_f():

    def forward(x,y):
        return torch.asinh(out=x, input=y)   
         
    x1 = torch.rand([10],dtype=torch.float32)
    x2 = x1.clone()
    y = torch.tensor([487875.875, -956238.8125, 630736.0, -161079.578125, 104060.9375, 757224.3125, -153601.859375, -648042.5, 733955.4375, -214764.90625],dtype=torch.float32)

    no_op_info = forward(x1,y)# result of eagermode
    print("eagermode results:", no_op_info)
    
    op_info = torch.compile(forward, mode='max-autotune-no-cudagraphs',fullgraph=True)(x2,y)# result of optimized mode
    print("optimized results:", op_info)
    
    assert not torch.allclose(no_op_info, op_info, rtol=1e-3, atol=1e-3, equal_nan=True)