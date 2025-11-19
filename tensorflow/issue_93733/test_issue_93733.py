import subprocess
import pytest

SIGILL_PROCESS_RETURNCODE = -4

def test_f():
    try:
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGILL_PROCESS_RETURNCODE
    else:
        assert False


test_f()