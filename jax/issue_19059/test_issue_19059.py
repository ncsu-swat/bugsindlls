import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

def g(hiddens):
    hiddens_aug = jnp.vstack((hiddens[0], hiddens))
    new_hiddens = hiddens_aug.copy()
    diff = new_hiddens[:-1] - hiddens
    diff = new_hiddens[:-1] - hiddens
    out = jnp.trace(jnp.conj(diff).T @ diff).real
    return jnp.array(out, dtype=jnp.complex64)


def f(carry, arg):
    primals, f_vjp = jax.vjp(
        g,
        jr.normal(jr.PRNGKey(0), (9, 8), dtype=jnp.complex64),
    )
    out = f_vjp(1.0 + 0j)[0]
    return carry, carry

def test_f():
    issue_no = '19059'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    with pytest.raises(TypeError) as e_info:
        a, b = jax.lax.scan(f, 0, jnp.arange(4, dtype=jnp.complex64))
    
    print(e_info.value)