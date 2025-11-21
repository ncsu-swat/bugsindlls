import torch
import numpy as np
import pytest
import torch.nn.functional as F


@pytest.mark.cuda
def test_conv_transpose2d_float16_cpu_gpu_diff():
    rng = np.random.default_rng(2186)

    input_tensor = torch.tensor(
        rng.uniform(-3.998, 5.0, (6, 7, 15, 23)),
        dtype=torch.float16
    )
    weight = torch.tensor(
        rng.uniform(-9990.0, -9990.0, (7, 3, 15, 18)),
        dtype=torch.float16
    )
    stride = 1
    padding = 2

    # ---------------- CPU ----------------
    cpu_failed = False
    out_cpu = None
    try:
        out_cpu = F.conv_transpose2d(input_tensor, weight,
                                     stride=stride, padding=padding)
    except RuntimeError:
        cpu_failed = True

    # ---------------- GPU ----------------
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU test")

    gpu_failed = False
    out_gpu = None
    try:
        out_gpu = F.conv_transpose2d(input_tensor.cuda(), weight.cuda(),
                                     stride=stride, padding=padding)
    except RuntimeError:
        gpu_failed = True

    # ---------------- EXPECTED BUG CONDITION ----------------
    # CPU fails, GPU succeeds → BUG reproduced → TEST PASSES
    if cpu_failed and not gpu_failed:
        print("Bug reproduced: CPU failure + GPU success")
        return  # PASS

    # ---------------- OTHERWISE → TEST FAILS ----------------
    pytest.fail(
        f"Bug did NOT reproduce.\n"
        f"cpu_failed={cpu_failed}, gpu_failed={gpu_failed}"
    )
