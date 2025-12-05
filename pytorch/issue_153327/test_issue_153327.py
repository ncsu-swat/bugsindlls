import subprocess
import pytest
import signal

def test_f():
    issue_no = '153327'
    print('Pytorch issue no.', issue_no)

    try:
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == -signal.SIGFPE
    else:
        assert False
