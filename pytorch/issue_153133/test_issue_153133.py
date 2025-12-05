import torch
import pytest

def test_f():

    print(torch.__version__)
    res = torch.tensor([])
    # This should fail on both CPU and GPU since it's an impossible range
    try:
        gpu_res = torch.arange(start=1549556900, end=1549556828, step=1989724, dtype=torch.float, device='cuda')
        print("GPU Results:", gpu_res)
        assert torch.equal(gpu_res.cpu(), res)
    except Exception as e:
        print("GPU Exception:", e)

    with pytest.raises(RuntimeError) as e_info:
        print("CPU Results:", torch.arange(start=1549556900, end=1549556828, step=1989724, dtype=torch.float, device='cpu'))
    print(f'{e_info.type.__name__}: {e_info.value}')
