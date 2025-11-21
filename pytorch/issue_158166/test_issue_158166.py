# test_lppool2d_cpu_gpu_diff.py
import torch
import pytest
import numpy as np
import torch.nn as nn

@pytest.mark.cuda
def test_lppool2d_cpu_gpu_diff():
    rng = np.random.default_rng(272)
    input_tensor = torch.tensor(rng.uniform(-1, 0, size=(1, 9, 1, 1)), dtype=torch.float64)
    norm_type = 2
    kernel_size = 1
    stride = np.iinfo(np.int32).max + 1
    ceil_mode = False

    # CPU behavior
    cpu_failed = False
    try:
        nn.LPPool2d(norm_type=norm_type, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)(input_tensor)
    except RuntimeError:
        cpu_failed = True

    # GPU behavior
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        nn.LPPool2d(norm_type=norm_type, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode).cuda()(input_tensor.cuda())
    except RuntimeError:
        gpu_failed = True

    # Pass test only if CPU and GPU behave differently
    assert cpu_failed != gpu_failed, "BUG NOT REPRODUCED: CPU/GPU difference not observed"
