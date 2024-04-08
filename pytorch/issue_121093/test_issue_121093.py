import subprocess
import pytest

SEGFAULT_PROCESS_RETURNCODE = -11

def test_f():
    try:     
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SEGFAULT_PROCESS_RETURNCODE
    else:
        assert False