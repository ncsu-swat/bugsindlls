import subprocess
import pytest

SIGSEGV_PROCESS_RETURNCODE = -11

def test_f():
    try:
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGSEGV_PROCESS_RETURNCODE
    else:
        assert False


test_f()