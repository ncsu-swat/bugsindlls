import subprocess
import pytest

SIGABRT_PROCESS_RETURNCODE = -6

def test_f():
    issue_no = '120762'
    print('Pytorch issue no.', issue_no)

    try:
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGABRT_PROCESS_RETURNCODE
    else:
        assert False