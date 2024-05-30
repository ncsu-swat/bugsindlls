import jax
import jax.random as rand
import os
import pytest

def test_f():
    issue_no = '17699'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    os.environ['JAX_PLATFORMS'] = 'cpu'

    jax.config.update('jax_enable_custom_prng', True)
    jax.config.update('jax_default_prng_impl', 'rbg')

    with pytest.warns(DeprecationWarning) as w_info:
        key = rand.PRNGKey(3407)
        print(isinstance(key, jax.random.PRNGKeyArray)) # Deprecation Warning: jax.random.PRNGKeyArray is deprecated.
        print(jax.dtypes.issubdtype(key, jax.dtypes.prng_key))
    print(f'{w_info[0].category.__name__}: {w_info[0].message}')
