import jax
from jax.scipy import stats as jstats
from scipy import stats
import numpy as np
import pytest

def test_f():
    issue_no = '17199'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    value = np.array(10, np.float32)
    res_scipy = stats.norm.sf(value)
    res_jax = jstats.norm.sf(value)

    print('scipy:', res_scipy)
    print('jax:', res_jax) # jax.scipy.stats.norm.sf is not accurate
