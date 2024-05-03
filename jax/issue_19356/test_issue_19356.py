import pytest
import os
from jax import numpy as jnp

def test_jax_numpy_methods_where_parameter():
    
    # Command to run command: pyright buggy_code.py and get the correponding error
    output = os.popen('pyright buggy_code.py').read()

    assert "error: No parameter named \"where\"" in output # The output shouldn't contain this error
