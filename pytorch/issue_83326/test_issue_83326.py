import torch
import pytest

def test_tril_cpu_gpu():
    print("Using torch", torch.__version__)

    # Random 1D tensor
    arg_1 = torch.rand([4], dtype=torch.float32)

    # GPU check
    if torch.cuda.is_available():
        arg_2 = arg_1.clone().cuda()
        res_gpu = torch.tril(arg_2)
        expected_gpu = torch.tensor([arg_2[0].item(), 0.0, 0.0, 0.0], device="cuda:0")
        print("GPU result:", res_gpu)
        print("Expected GPU:", expected_gpu)
        assert torch.allclose(res_gpu, expected_gpu, atol=1e-4)

    # CPU check (should raise IndexError for 1D input)
    with pytest.raises(IndexError) as e_info:
        torch.tril(arg_1)
    print(f"{e_info.type.__name__}: {e_info.value}")
