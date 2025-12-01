import torch
import pytest

def test_f():

    print(torch.__version__)

    # Create tensors
    x1 = torch.tensor(-1.6437e+10, dtype=torch.float64)  # Large value in float64
    x2 = torch.full((3, 8, 2, 6), float('nan'), dtype=torch.float16)  # NaN values in float16

    # CPU implementation correctly raises an error
    with pytest.raises(RuntimeError) as e_info:
        res = torch.cosine_similarity(x1, x2, dim=0, eps=-9.74982e+23)
        print("CPU result:", res)
    print(f'{e_info.type.__name__}: {e_info.value}')

    # CUDA implementation silently produces NaN values
    x1_cuda = x1.cuda()
    x2_cuda = x2.cuda()
    try:
        res = torch.cosine_similarity(x1_cuda, x2_cuda, dim=0, eps=-9.74982e+23)
        print("CUDA result:", res)
        assert True
    except RuntimeError as e:
        print("CUDA error:", e)
