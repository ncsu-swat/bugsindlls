import pytest
import torch

def test_f1(): 
    tensor = torch.rand([2, 2])
    res_relu=torch.nn.functional.relu(tensor, "aaaa")
    print(res_relu)
    assert res_relu is not None

def test_f2(): 
    tensor = torch.rand([2, 2])
    max_pool1d_res=torch.nn.functional.max_pool1d(tensor, 1, return_indices="aaa")
    print(max_pool1d_res)
    assert max_pool1d_res is not None