import pytest
import torch
import torch.nn.functional as F

def test_f():

    a = torch.tensor([[0, 1], [2, 3]])
    for i in [0, 1, 2]:
        assert(torch.equal(torch.diag(a, i), torch.diagonal(a, i)))
    res=torch.diagonal(a, 3)
    print(res)
    with pytest.raises(RuntimeError) as e_info:
        diag_res=torch.diag(a, 3)
        print(diag_res)
        # RuntimeError: [enforce fail at CPUAllocator.cpp:50] ((ptrdiff_t)nbytes) >= 0. alloc_cpu() seems to have been called with negative number: 18446744073709551608
    print(f"{e_info.type.__name__}: {e_info.value}")