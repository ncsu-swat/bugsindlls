import pytest
import torch

def test_f():
    a = torch.rand([0, 3])
    amax_res=torch.amax(a)
    print(amax_res)
    # tensor(-1.8891e+26)
    amin_res=torch.amin(a)
    print(amin_res)
    # tensor(9.1477e-41)
    assert amax_res is not None
    assert amin_res is not None