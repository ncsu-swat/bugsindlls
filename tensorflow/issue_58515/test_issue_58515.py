import subprocess
import pytest


expected_output = """tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]], shape=(3, 3), dtype=float64) tf.Tensor(
[[ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0. -0.]], shape=(3, 3), dtype=float64) tf.Tensor(
[[ inf  inf  inf]
 [ inf  inf  inf]
 [ inf  inf -inf]], shape=(3, 3), dtype=float64)
"""

def test_f():
    result = subprocess.run(
        ["python3", "-m", "buggy_code"],
        capture_output=True,
        text=True
    )

    # Strip trailing newlines/spaces before comparing
    actual_output = result.stdout.strip()
    expected = expected_output.strip()

    assert actual_output == expected
