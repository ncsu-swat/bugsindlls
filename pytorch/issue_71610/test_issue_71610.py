import pytest
import torch

def test_f():
    input = torch.rand([2, 0, 5, 5], dtype=torch.complex128)
    A = torch.rand([1, 5, 5], dtype=torch.complex128)

    with pytest.raises(RuntimeError) as e_info:
        torch.linalg.solve(A, input)
    print(f"{e_info.type.__name__}: {e_info.value}")