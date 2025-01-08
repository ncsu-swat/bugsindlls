import torch
from torch.func import jacfwd
import pytest

def func(x):
    two = 2.0
    return two * x

def jac_func(x):
    return jacfwd(func, argnums=(0,))(x)

def test_f():
    issue_no = '125078'
    print('Pytorch issue no.', issue_no)

    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed) as e_info:
        compiled_jac_func = torch.compile(jac_func)
        compiled_jac_func(torch.ones((3,), dtype=torch.float64))
    print(f'{e_info.type.__name__}: {e_info.value}')
