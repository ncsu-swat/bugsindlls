import pytest
import torch

def test_f():
    # 71610
    input_tensor = torch.rand([2, 0, 5, 5], dtype=torch.complex128)
    A = torch.rand([1, 5, 5], dtype=torch.complex128)

    with pytest.raises(RuntimeError) as excinfo:
        torch.linalg.solve(A, input_tensor)

    assert "INTERNAL ASSERT FAILED" in str(excinfo.value)