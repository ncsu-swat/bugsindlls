# test_fmin_cpu_gpu_diff.py
import torch
import pytest

@pytest.mark.cuda
def test_fmin_cpu_gpu_diff():
    input_tensor = torch.tensor([-2000000], dtype=torch.int64)
    other = torch.tensor([-1])
    output_cpu = torch.tensor([], dtype=torch.int16)
    output_gpu = torch.tensor([], dtype=torch.int16).cuda()

    # CPU behavior
    cpu_failed = False
    try:
        torch.fmin(input_tensor, other, out=output_cpu)
    except RuntimeError:
        cpu_failed = True

    # GPU behavior
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        torch.fmin(input_tensor.cuda(), other.cuda(), out=output_gpu)
        if not torch.equal(output_cpu, output_gpu.cpu()):
            gpu_failed = True
    except RuntimeError:
        gpu_failed = True

    # Pass test only if CPU and GPU differ
    assert cpu_failed != gpu_failed, "BUG NOT REPRODUCED: CPU/GPU difference not observed"
