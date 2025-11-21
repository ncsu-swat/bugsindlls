# test_softshrink_cpu_gpu_diff.py
import torch
import pytest
import torch.nn.functional as F

@pytest.mark.cuda
def test_softshrink_cpu_gpu_diff():
    input_tensor = torch.randn(34, 43, dtype=torch.float16)
    lambd = 65507.0

    # CPU behavior
    cpu_failed = False
    try:
        F.softshrink(input_tensor, lambd)
    except RuntimeError:
        cpu_failed = True

    # GPU behavior
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        F.softshrink(input_tensor.cuda(), lambd)
    except RuntimeError:
        gpu_failed = True

    # Pass test only if CPU and GPU differ
    assert cpu_failed != gpu_failed, "BUG NOT REPRODUCED: CPU/GPU difference not observed"
