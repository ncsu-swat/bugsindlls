import subprocess
import pytest
import signal

def test_f():
<<<<<<<< HEAD:pytorch/issue_94669/test_issue_94669.py
    issue_no = '94669'
========
    issue_no = '97087'
>>>>>>>> origin/main:pytorch/issue_97087/test_issue_97087.py
    print('Pytorch issue no.', issue_no)

    try:
        subprocess.run(["python3", "-m", "buggy_code.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == -signal.SIGSEGV    # Process returns SIGSEGV signal (segmentation fault)
    else:
        assert False