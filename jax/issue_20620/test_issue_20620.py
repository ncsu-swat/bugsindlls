import pytest
import jax
import numpy as np

def test_f():
    issue_no = '20620'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    assert isinstance(np.zeros(10, jax.float0), np.ndarray) is True # Should be False rather than True
