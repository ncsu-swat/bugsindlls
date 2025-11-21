import torch
import torch.nn.functional as F
import pytest
import numpy as np

@pytest.mark.cuda
def test_conv_transpose2d_cpu_gpu_diff():
    torch.backends.cudnn.deterministic = True

    rng = np.random.default_rng(1146)
    input_tensor = torch.tensor(rng.uniform(-1, 13, (9, 9, 20, 52)), dtype=torch.float16)
    weight = torch.tensor(rng.uniform(-2, -1, (9, 11, 12, 84)), dtype=torch.float16)
    stride = 1
    padding = 12

    # CPU output
    cpu_failed = False
    try:
        out_cpu = F.conv_transpose2d(input_tensor, weight, stride=stride, padding=padding)
    except RuntimeError:
        cpu_failed = True

    # GPU output
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        out_gpu = F.conv_transpose2d(input_tensor.cuda(), weight.cuda(), stride=stride, padding=padding)
    except RuntimeError:
        gpu_failed = True

    # Pass test only if CPU fails differently than GPU
    assert (cpu_failed != gpu_failed), "BUG NOT REPRODUCED: CPU/GPU difference not observed"

    # Optionally compare outputs if both succeeded
    if not cpu_failed and not gpu_failed:
        assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-3), "CPU/GPU outputs differ unexpectedly"
