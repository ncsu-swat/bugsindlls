import torch
import pytest


def f():
    p = torch.polar(torch.ones_like(torch.empty(2, 3)), torch.empty(2, 3))
    p.index_select(0, torch.arange(0, 2, dtype=torch.int64).to("mps"))


def test_f():
    issue_no = '120803'
    print('PyTorch issue no.', issue_no)

    # Check the MPS support
    if not torch.backends.mps.is_available():
        pytest.skip('MPS is not available')
    
    with pytest.raises(RuntimeError) as e_info:
        f()
    
    print(e_info.value)