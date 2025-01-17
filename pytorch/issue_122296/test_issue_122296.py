import torch
import pytest

def foo(x):
    a = x.item()
    torch._constrain_as_size(a, min=1, max=10)
    return torch.ones(a, a)

def test_f():
    torch.utils.collect_env.main()
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed) as e_info:
        torch._dynamo.config.capture_scalar_outputs = True
        fn = torch.compile(foo, fullgraph=True, dynamic=True)
        fn(torch.tensor(5))
    print(f'{e_info.type.__name__}: {e_info.value}')