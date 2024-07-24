import jax
import jax.numpy as jnp
import pytest

def run(some_tracer):
    def f(x, y):
        return x + y

    g = lambda x: f(x, some_tracer)
    jaxpr = jax.make_jaxpr(g)(1)
    jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1)

def test_run():
    issue_no = '21116'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    with pytest.raises(ValueError) as e_info:
        jax.vmap(run)(jnp.arange(2)) # ValueError: safe_map() argument 2 is shorter than argument 1
    print(f'{e_info.type.__name__}: {e_info.value}')
