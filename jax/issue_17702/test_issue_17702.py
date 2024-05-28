import jax
import numpy as np
import jax.numpy as jnp
import os, psutil
import pytest

def test_f():
    issue_no = '17922'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()


    process = psutil.Process(os.getpid())

    baseline_memory = process.memory_info().rss / 1e9  # 0.1 GB
    x = np.random.uniform(0, 1, (10000, 20000)).astype("float32")
    memory_np = process.memory_info().rss / 1e9  # 0.9 GB
    y = jnp.asarray(x)
    memory_jax = process.memory_info().rss / 1e9  # 1.7 GB

    tolerance = 0.1

    expected_increase = memory_np - baseline_memory  
    actual_increase = memory_jax - memory_np
    
    assert actual_increase > expected_increase + tolerance # memory should increase by same amount, but does increase

