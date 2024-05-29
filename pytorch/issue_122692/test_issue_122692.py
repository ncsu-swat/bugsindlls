import subprocess
import os
import pytest

def test_f():
    issue_no = '122692'
    print('Pytorch issue no.', issue_no)

    env = os.environ.copy()
    env['TORCH_COMPILE_DEBUG'] = '1'
    result = subprocess.run(["python3", "buggy_code.py"], check=True, env=env, capture_output=True, text=True)
    print(result.stderr)

    error = "torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor bool call_method is_complex"
    assert error in result.stderr # Error is present in stderr logs
