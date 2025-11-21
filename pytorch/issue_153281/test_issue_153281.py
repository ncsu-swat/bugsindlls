# test_rrelu_cpu_gpu_diff.py
import torch
import pytest
import torch.nn.functional as F

@pytest.mark.cuda
def test_rrelu_cpu_gpu_difference():
    input_tensor = torch.tensor([-1.0])
    lower = 0
    upper = float('inf')

    # CPU behavior
    cpu_failed = False
    try:
        F.rrelu(input_tensor, lower, upper, training=True)
    except RuntimeError:
        cpu_failed = True

    # GPU behavior
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        F.rrelu(input_tensor.cuda(), lower, upper, training=True)
    except RuntimeError:
        gpu_failed = True

    # Pass test if CPU and GPU behave differently
    assert cpu_failed != gpu_failed, "BUG NOT REPRODUCED: CPU/GPU difference not observed"
