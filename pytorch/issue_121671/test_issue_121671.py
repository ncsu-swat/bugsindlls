import torch
import pytest


def f():
    x = torch.tensor([[1, 2], [3, 4]], device='cpu').to_sparse_csr()
    torch.empty_like(x, device='cuda')


def test_f():
    issue_no = '121671'
    print('PyTorch issue no.', issue_no)

    with pytest.raises(RuntimeError) as e_info:
        f()
    
    print(f'{e_info.type.__name__}: {e_info.value}')