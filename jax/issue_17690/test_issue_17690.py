import jax
import jax.numpy as jnp
import os
import pytest

def test_f():
    issue_no = '17690'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    os.environ['XLA_FLAGS'] = (
      f'{os.environ.get("XLA_FLAGS", "")} '
      '--xla_force_host_platform_device_count=4'
        )

    with pytest.raises(ValueError) as e_info:
        x = jax.pmap(lambda x: x, in_axes=0, out_axes=None)(jnp.arange(jax.device_count()))
        jnp.array(x) # ValueError as pmap must have at least one non-None value in in_axes
    print(f'{e_info.type.__name__}: {e_info.value}')
