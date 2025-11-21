# test_det_complex128_cpu_gpu_diff.py
import torch
import pytest
import numpy as np

@pytest.mark.cuda
def test_det_complex128_cpu_gpu_diff():
    rng = np.random.default_rng(716)
    input_tensor = torch.tensor(
        rng.uniform(-9223320268019550000., -9223320268019550000., size=(1, 1, 1, 1, 3, 3)),
        dtype=torch.complex128
    )

    # CPU behavior
    cpu_failed = False
    try:
        out_cpu = torch.det(input_tensor)
    except RuntimeError:
        cpu_failed = True

    # GPU behavior
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        out_gpu = torch.det(input_tensor.to("cuda"))
        # Compare CPU vs GPU
        if not torch.allclose(out_cpu, out_gpu.cpu(), rtol=1e-12, atol=1e-12):
            gpu_failed = True
    except RuntimeError:
        gpu_failed = True

    # Pass test only if CPU and GPU behave differently
    assert cpu_failed != gpu_failed, "BUG NOT REPRODUCED: CPU/GPU difference not observed"
