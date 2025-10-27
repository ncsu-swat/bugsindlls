import subprocess
import pytest

SIGFPE_PROCESS_RETURNCODE = -8

def test_f():
    issue_no = '121092'
    print('Pytorch issue no.', issue_no)

    try:
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGFPE_PROCESS_RETURNCODE
    else:
        assert False
