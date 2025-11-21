# test_celu_large_alpha_cpu_gpu_diff.py
import torch
import pytest
import torch.nn.functional as F

@pytest.mark.cuda
def test_celu_large_alpha_cpu_gpu_diff():
    input_tensor = torch.ones(32, 2, dtype=torch.float64) * -11.
    alpha = 324112638312866870.

    # CPU behavior
    cpu_failed = False
    try:
        out_cpu = F.celu(input_tensor, alpha=alpha)
    except RuntimeError:
        cpu_failed = True

    # GPU behavior
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        out_gpu = F.celu(input_tensor.cuda(), alpha=alpha)
        # Compare CPU vs GPU
        if not torch.allclose(out_cpu, out_gpu.cpu(), rtol=1e-12, atol=1e-12):
            gpu_failed = True
    except RuntimeError:
        gpu_failed = True

    # Pass only if CPU and GPU behave differently
    assert cpu_failed != gpu_failed, "BUG NOT REPRODUCED: CPU/GPU difference not observed"
