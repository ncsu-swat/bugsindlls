import pytest
import torch
import sys

def test_f():
    
    arg_1 = torch.tensor([0,1,2,3,4])
    arg_2 = 6
    res1 = torch.kthvalue(arg_1,arg_2,)
    res3 = torch.kthvalue(arg_1,arg_2,)
    res2 = arg_1.kthvalue(arg_2,)
    print(res1)
    print(res2)
    print(res3)
    assert res1 is not None