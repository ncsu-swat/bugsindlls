import subprocess
import pytest

SIGABRT_PROCESS_RETURNCODE = -6

def test_f1():
    try:
        subprocess.run(["python3", "-m", "buggy_code1.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGABRT_PROCESS_RETURNCODE
    else:
        assert False
def test_f2():
    try:
        subprocess.run(["python3", "-m", "buggy_code2.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGABRT_PROCESS_RETURNCODE
    else:
        assert False
