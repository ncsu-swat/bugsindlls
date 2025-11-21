# test_floor_divide_cpu_gpu_diff.py
import torch
import pytest
import numpy as np

@pytest.mark.cuda
def test_floor_divide_cpu_gpu_diff():
    rng = np.random.default_rng(1371)

    input_tensor = torch.tensor(rng.uniform(7, 7, (7, 8, 1, 1, 8, 7)), dtype=torch.int8)
    other_tensor = torch.tensor(rng.uniform(-1, 7, (7, 8, 8, 8, 8, 7)), dtype=torch.float16)

    # CPU output
    cpu_failed = False
    try:
        out_cpu = torch.floor_divide(input_tensor, other_tensor)
    except RuntimeError:
        cpu_failed = True

    # GPU output
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        out_gpu = torch.floor_divide(input_tensor.cuda(), other_tensor.cuda())
        # Compare CPU vs GPU
        if not torch.allclose(out_cpu, out_gpu.cpu(), rtol=1e-7, atol=1e-2):
            gpu_failed = True
    except RuntimeError:
        gpu_failed = True

    # Pass test only if CPU and GPU differ
    assert cpu_failed != gpu_failed, "BUG NOT REPRODUCED: CPU/GPU difference not observed"
