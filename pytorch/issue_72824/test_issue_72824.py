import pytest
import torch
import torch.nn as nn

def test_f1():

    results = dict()
    a = torch.rand([3], dtype=torch.float32)
    b = torch.rand([3], dtype=torch.float64)
    res=torch.nn.BCEWithLogitsLoss()(a, b)
    print(res)
    assert res is not None

    with pytest.raises(RuntimeError) as e_info:
        torch.nn.BCELoss()(a, b)
    print(f"{e_info.type.__name__}: {e_info.value}")

def test_f2():

    results = dict()
    a = torch.rand([3,4], dtype=torch.float32)
    b = torch.rand([3,4], dtype=torch.float64)
    res=torch.nn.MultiLabelSoftMarginLoss()(a, b)
    print(res)
    assert res is not None

def test_f3():

    results = dict()
    a = torch.rand([3], dtype=torch.float32)
    b = torch.rand([3], dtype=torch.float64)
    res=torch.nn.PoissonNLLLoss()(a, b)
    print(res)
    assert res is not None