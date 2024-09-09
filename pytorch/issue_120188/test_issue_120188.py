import torch
from torch.nested._internal.nested_tensor import jagged_from_list
import pytest

def f(x, y):
    z = x.sin()
    y.sin_()
    return z.cos(), y.cos()

def test_f():
    issue_no = '120188'
    print('Pytorch issue no.', issue_no)

    f_c = torch.compile(f, backend="inductor")

    values = [torch.rand((i, 8), requires_grad=True) for i in range(1, 6)]
    values_copy = [x.detach().clone().requires_grad_(True) for x in values]

    nt, offsets = jagged_from_list(values, None)
    nt_copy, offsets = jagged_from_list(values_copy, offsets)
    y = torch.rand((4, 8))
    y_copy = y.clone()
    
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed) as e_info:
        ret = f_c(nt, y)[0]
        ref = f(nt_copy, y_copy)[0]
    print(f'{e_info.type.__name__}: {e_info.value}')
