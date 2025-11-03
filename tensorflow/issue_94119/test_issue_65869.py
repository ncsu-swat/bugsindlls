import subprocess
import pytest

SIGABRT_PROCESS_RETURNCODE = -6

def test_f():
    try:
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGABRT_PROCESS_RETURNCODE
    else:
        assert False


test_f()