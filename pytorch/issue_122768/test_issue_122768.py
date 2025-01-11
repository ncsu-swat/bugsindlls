import torch
from torch._dynamo.exc import TorchRuntimeError
import triton
import pytest

@triton.jit
def test_kernel(kernel):
    pass

@torch.compile(backend="eager")
def f(x):
    grid = (x.numel(),)
    # test_kernel[grid](kernel=x)
    with torch.cuda.device(x.device.index):
        test_kernel[grid](kernel=x)

def test_f():
    issue_no = '122768'
    print('PyTorch issue no.', issue_no)

    t1 = torch.rand(5, device="cuda")
    with pytest.raises(TorchRuntimeError) as e_info:
        f(t1) # Raises TorchRuntimeError
    print(f'{e_info.type.__name__}: {e_info.value}')
