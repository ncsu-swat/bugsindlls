import pytest
import torch

def test_f1():

    input = torch.rand([], dtype=torch.float64)
    dim = 100
    cummin_res=torch.cummin(input, dim)
    print("cummin result:", cummin_res)
    assert cummin_res is not None
    
def test_f2():

    input = torch.rand([], dtype=torch.float64)
    dim = 100
    cummax_res=torch.cummax(input, dim)
    print("cummax result:", cummax_res)
    assert cummax_res is not None
    
def test_f3():

    input = torch.rand([], dtype=torch.float64)
    dim = 100
    sort_res=torch.sort(input, dim)
    print("sort result:", sort_res)
    assert sort_res is not None
    
def test_f4():

    input = torch.rand([], dtype=torch.float64)
    dim = 100
    argsort_res=torch.argsort(input, dim)
    print("argsort result:", argsort_res)
    assert argsort_res is not None