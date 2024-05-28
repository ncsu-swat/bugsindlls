import torch
import pytest


def f():
    x = torch.zeros(3, device='mps')
    x[1] = 1
    x[2] = 3
    return x

def test_f():
    issue_no = '122016'
    print('PyTorch issue no.', issue_no)

    # Check the MPS support
    if not torch.backends.mps.is_available():
        pytest.skip('MPS is not available')
    
    result = f()

    assert result[1] == 3  # Work, but should not work since x[1] = 1
