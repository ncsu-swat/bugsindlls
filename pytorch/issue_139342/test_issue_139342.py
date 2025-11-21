# test_logcumsumexp_cpu_gpu_diff.py
import torch
import pytest

@pytest.mark.cuda
def test_logcumsumexp_cpu_gpu_diff():
    scalar_input = torch.tensor(22, dtype=torch.int64)
    tensor_input = torch.tensor([22], dtype=torch.int64)

    # CPU behavior
    cpu_scalar_failed = False
    cpu_tensor_failed = False
    try:
        torch.logcumsumexp(scalar_input, 0)
    except RuntimeError:
        cpu_scalar_failed = True
    try:
        torch.logcumsumexp(tensor_input, 0)
    except RuntimeError:
        cpu_tensor_failed = True

    # GPU behavior
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_scalar_failed = False
    gpu_tensor_failed = False
    try:
        torch.logcumsumexp(scalar_input.cuda(), 0)
    except RuntimeError:
        gpu_scalar_failed = True
    try:
        torch.logcumsumexp(tensor_input.cuda(), 0)
    except RuntimeError:
        gpu_tensor_failed = True

    # Pass only if CPU fails differently than GPU
    assert (cpu_scalar_failed != gpu_scalar_failed) or (cpu_tensor_failed != gpu_tensor_failed), \
        "BUG NOT REPRODUCED: CPU/GPU difference not observed"
