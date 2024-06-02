import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.sparse import BCOO
import pytest

def test_f():
    issue_no = '17448'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    

    i = jnp.int32(1)
    # can fix issue with BCOO by casting to a python int dtype
    # i = int(i) 

    array_shape = (i,)
    # m_dense = jnp.ones(shape=array_shape)
    m_sparse = BCOO((jnp.array([1]), jnp.array([[0]])), shape=array_shape)

    # Should be the same?
    # print(m_dense.shape) # --> (1,)
    # print(m_sparse.shape) # --> (Array(1, dtype=int32),)
    # print(m_sparse.todense().shape) # --> (1,)
    with pytest.raises(jax.errors.TracerBoolConversionError) as e_info:
        jit(lambda _: m_sparse[0])(None) # Throws TracerBoolConversionError

    print(e_info.value)

test_f()