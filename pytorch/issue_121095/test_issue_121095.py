import subprocess
import pytest

SIGSEG_PROCESS_RETURNCODE = -11

def test_f():
    try:     
        subprocess.run(["python3", "-m", "buggy_code"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGSEG_PROCESS_RETURNCODE
    else:
        assert False