import pytest
import torch
import torch.nn as nn

def test_f1():
    
    a = torch.rand([1, 0, 2])
    batchnorm = torch.nn.BatchNorm1d(5)
    res=batchnorm(a)
    print("BatchNorm1d Result:", res)
    assert res is not None

def test_f2():
    
    a = torch.rand([1, 0, 2, 2])
    batchnorm = torch.nn.BatchNorm2d(5)
    res=batchnorm(a)
    print("BatchNorm2d Result:", res)
    assert res is not None  

def test_f3():
    
    a = torch.rand([1, 0, 3, 2, 2])
    batchnorm = torch.nn.BatchNorm3d(5)
    res=batchnorm(a)
    print("BatchNorm3d Result:", res)
    assert res is not None

def test_f4():
    
    a = torch.rand([1, 0, 2])
    instanceNorm = torch.nn.InstanceNorm1d(5)
    res=instanceNorm(a)
    print("InstanceNorm1d Result:", res)
    assert res is not None

def test_f5():
    
    a = torch.rand([1, 0, 2, 2])
    instanceNorm = torch.nn.InstanceNorm2d(5)
    res=instanceNorm(a)
    print("InstanceNorm2d Result:", res)
    assert res is not None

def test_f6():
    
    a = torch.rand([1, 0, 3, 2, 2])
    instanceNorm = torch.nn.InstanceNorm3d(5)
    res=instanceNorm(a)
    print("InstanceNorm3d Result:", res)
    assert res is not None