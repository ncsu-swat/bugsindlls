import subprocess
import sys
import pytest
import os


@pytest.mark.cuda
def test_native_channel_shuffle_cpu_gpu_diff():
    """
    PASS = bug reproduced (Python crash in subprocess)
    FAIL = bug NOT reproduced (subprocess exits normally)
    """

    test_script = r"""
import torch
import torch.nn.functional as F

# Input that triggers CPU crash
x = torch.randn((1, 4, 2, 2), dtype=torch.float32)
groups = 0  # Invalid (forces division by zero inside ATen)
y = torch.channel_shuffle(x, groups)
"""

    # Write script to temp file
    with open("run_crash_test.py", "w") as f:
        f.write(test_script)

    # Run risky PyTorch code in subprocess so pytest doesn't die
    result = subprocess.run(
        [sys.executable, "run_crash_test.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # A floating-point exception crash returns non-zero exit code
    crashed = result.returncode != 0

    if crashed:
        print("Bug reproduced: subprocess crashed with floating point exception")
        return  # PASS

    pytest.fail("Bug did NOT reproduce — subprocess exited normally")
