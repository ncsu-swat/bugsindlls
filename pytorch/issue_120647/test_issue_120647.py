import torch
import pytest

@torch.compile()
def f(x, y):
    code = compile(y, "f", "exec")
    exec(y)
    return x

def test_f():
    issue_no = '120647'
    print('Pytorch issue no.', issue_no)

    with pytest.raises(torch._dynamo.exc.InternalTorchDynamoError) as e_info:
        print(f(torch.rand(3), "print('Hello World')"))
    print(f'{e_info.type.__name__}: {e_info.value}')
