# test_normalize_complex_cpu_gpu_repro.py
import torch
import pytest
import numpy as np

@pytest.mark.cuda
def test_normalize_complex_cpu_gpu_difference():
    # Large p and small magnitude to trigger CPU NaNs
    p = 196
    a = torch.tensor([
        [1e-5 + 0.j, 2e-5 + 0.j],
        [3e-5 + 0.j, 4e-5 + 0.j],
        [5e-5 + 0.j, 6e-5 + 0.j],
        [7e-5 + 0.j, 8e-5 + 0.j],
        [9e-5 + 0.j, 1e-4 + 0.j],
    ], dtype=torch.complex64)

    # CPU: this should produce NaNs
    cpu_output = torch.nn.functional.normalize(a, p=p)
    cpu_has_nan = torch.isnan(cpu_output).any()

    # GPU: should not produce NaNs
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU part")
    gpu_output = torch.nn.functional.normalize(a.cuda(), p=p)
    gpu_has_nan = torch.isnan(gpu_output).any()

    # Test passes only if CPU has NaNs but GPU does not
    assert cpu_has_nan and not gpu_has_nan, "BUG NOT REPRODUCED: CPU did not produce NaNs or GPU matches CPU"
