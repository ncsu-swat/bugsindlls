import torch
import pytest

@torch.jit.script_if_tracing
def g(x, y, _ = None):
    return torch.cos(x * y)

@torch.compile(backend="eager")
def f(x, z):
    return g(x, 2, z)

def test_f():
    issue_no = '120387'
    print('Pytorch issue no.', issue_no)

    with pytest.raises(torch._dynamo.exc.InternalTorchDynamoError) as e_info:
        print(f(torch.randn(4), 1))
        print(f(torch.randn(4), torch.randn(4)))
    print(f'{e_info.type.__name__}: {e_info.value}')
