import subprocess
import pytest
import signal

def test_f1():
    try:
        subprocess.run(["python3", "-m", "buggy_code1.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == -signal.SIGFPE  
    else:
        assert False

def test_f2():
    try:
        subprocess.run(["python3", "-m", "buggy_code2.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == -signal.SIGFPE  
    else:
        assert False

def test_f3():
    try:
        subprocess.run(["python3", "-m", "buggy_code3.py"], check=True)
    except subprocess.CalledProcessError as err:
        print(err)
        assert err.returncode == -signal.SIGFPE 
    else:
        assert False
