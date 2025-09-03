import subprocess
import pytest

SIGABRT_PROCESS_RETURNCODE = -6


def test_adjust_hue_subprocess():
    try:
        subprocess.run( ["python3", "buggy_code.py"], check=True, timeout=5)
    except subprocess.CalledProcessError as err:
        pytest.fail(f"Process crashed with exit code {err.returncode}")
    except subprocess.TimeoutExpired:
        assert True
    else:
        pytest.fail("Process exited normally (bug not reproduced)")
test_adjust_hue_subprocess()