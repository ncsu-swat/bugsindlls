import subprocess
import sys
import tensorflow


SIGABRT_PROCESS_RETURNCODE = -6  # Expected return code for SIGABRT

def test_f():
    try:
        # Use module name, not filename
        subprocess.run([sys.executable, "-m", "buggy_code"], check=True)
    except subprocess.CalledProcessError as err:
        print(f"Subprocess failed with return code: {err.returncode}")
        # Fail if the bug did NOT produce SIGABRT
        assert err.returncode == SIGABRT_PROCESS_RETURNCODE, (
            f"Expected SIGABRT (-6), got {err.returncode}"
        )
    else:
        # If it ran without crashing, that is a test failure
        assert False, "buggy_code did NOT crash as expected"


if __name__ == "__main__":
    test_f()
