import torch
import pytest

def test_f():

    # Works fine on GPU
    try:
        gpu_res=torch.randn(2, 4, 15, dtype=torch.complex128, device='cuda').nanmean()
        print("GPU Results:", gpu_res)
        assert True
    except Exception as e:
        print("GPU Exception:", e)

    # Fails on CPU with misleading message
    with pytest.raises(RuntimeError) as e_info:
        print("CPU Results:", torch.randn(2, 4, 15, dtype=torch.complex128, device='cpu').nanmean())
    print(f'{e_info.type.__name__}: {e_info.value}')
