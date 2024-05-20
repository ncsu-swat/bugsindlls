import subprocess
import pytest

SIGABRT_PROCESS_RETURNCODE = -11

def test_f():
    issue_no = '121188'
    print('Pytorch issue no.', issue_no)

    try:
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == SIGABRT_PROCESS_RETURNCODE # Process returns SIGSEGV signal (segmentation fault)
    else:
        assert False
