import torch
import pytest


def f():
    return torch.backends.openmp.is_available()

def test_f():
    issue_no = '122705'
    print('PyTorch issue no.', issue_no)
    
    result = f()

    assert result == False  # Work, but should not work since result should be True
