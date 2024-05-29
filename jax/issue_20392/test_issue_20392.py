import os

num_devices = 8

os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices}'


import jax
import pytest

import warnings
warnings.filterwarnings("ignore", message="The jitted function pmap_batched_fn includes a pmap.*")

def single_fn(key):
    return jax.random.normal(key, ())

def pmap_batched_fn(key):
    batch_size = num_devices
    keys = jax.random.split(key, batch_size)
    pmap_fn = jax.pmap(single_fn)
    return pmap_fn(keys)
    
@pytest.mark.filterwarnings("ignore")
def test_f():
    issue_no = '20392'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    key = jax.random.key(0) # Some rogue jax.random.key, works with PRNGkey however


    with pytest.raises(ValueError) as e_info:
        jax.jit(pmap_batched_fn)(key)
    
    print(f'{e_info.type.__name__}: {e_info.value}')
