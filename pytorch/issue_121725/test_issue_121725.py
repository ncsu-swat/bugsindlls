import torch
import pytest

def test_f():
    issue_no = '121725'
    print('Pytorch issue no.', issue_no)

    with pytest.raises(RuntimeError) as e_info:
        print(torch.logsumexp(torch.randn(3, 3), dim=(0, 1))) # tensor(2.4638)
        print(torch.logsumexp(torch.randn(3, 3), dim=None))
    print(f'{e_info.type.__name__}: {e_info.value}')
