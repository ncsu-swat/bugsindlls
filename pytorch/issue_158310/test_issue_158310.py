import pytest
import torch
import torch.nn as nn
import numpy as np

@pytest.mark.cuda
def test_logsoftmax_float16_cpu_gpu_diff():
    rng = np.random.default_rng(432)
    input_tensor = torch.tensor(rng.uniform(-20260., -19740., size=(18, 1, 2, 2, 21, 2)), dtype=torch.float16)
    dim = 2

    # ---------------- CPU ----------------
    cpu_failed = False
    try:
        out_cpu = nn.LogSoftmax(dim=dim)(input_tensor)
    except RuntimeError:
        cpu_failed = True

    # ---------------- GPU ----------------
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    try:
        out_gpu = nn.LogSoftmax(dim=dim).cuda()(input_tensor.cuda())
    except RuntimeError:
        gpu_failed = True

    # ---------------- Test logic ----------------
    if cpu_failed != gpu_failed:
        # BUG reproduced → test PASSES
        pass
    else:
        # CPU and GPU behave the same → bug NOT reproduced → fail test
        pytest.fail("BUG NOT REPRODUCED: CPU and GPU behaved the same")
