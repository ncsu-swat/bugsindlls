import jax
import jax.numpy as jnp
import pytest

def f(xx):
    """twice all_to_all is expected to give an identical output as input"""
    yy = jax.lax.all_to_all(xx, "p", 1, 0)
    zz = jax.lax.all_to_all(yy, "p", 0, 1)
    return zz

def test_f():
    issue_no = '18122'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    pa2a = jax.pmap(f, axis_name="p", in_axes=0, out_axes=0)
    # test on real array
    aa = jnp.arange(2*3*2*4).reshape(2,3,2,4)
    bb = pa2a(aa)
    print('diff in real case', jnp.max(jnp.abs(aa-bb)))
    # test on complex array
    aa = jnp.arange(2*3*2*4).reshape(2,3,2,4)
    aa = aa + 1.J*aa
    bb = pa2a(aa)
    print('diff in complex case', jnp.max(jnp.abs(aa-bb)))
