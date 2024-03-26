import torch
import torch._dynamo
import pytest

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile()
def f(sz, x):
    s0, s1 = sz.tolist()
    r0, r1 = torch.ops.aten.split_with_sizes.default(x, [s0, s1])
    return torch.ops.aten.sort.default(r1)

def test_f():
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed) as e_info:
        N = 7312
        S0 = 420
        S1 = N - S0

        f(torch.tensor([S0, S1]), torch.randn(N))
    print(e_info)