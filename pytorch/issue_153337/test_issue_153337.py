import subprocess
import pytest

def test_f():
    try:
        subprocess.run( ["python3", "buggy_code.py"], check=True, timeout=120)
    except subprocess.CalledProcessError as err:
        pytest.fail(f"Process crashed with exit code {err.returncode}")
    except subprocess.TimeoutExpired:
        assert True
    else:
        pytest.fail("Process exited normally (bug not reproduced)")