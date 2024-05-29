import jax
import jax.numpy as jnp
import pytest

def scalar_add(x, y):
  # emphasize that only scalar tracers will be passed to this function.
  assert jnp.shape(x) == jnp.shape(y) == ()
  return x + y

def f():
  add = jnp.frompyfunc(scalar_add, nin=2, nout=1, identity=0)

  x = jnp.ones((5, 3))
  indices = jnp.array([0,4,2])
  t = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

  x = add.at(x, indices, t, inplace=False)
  # ValueError: safe_zip() argument 2 is shorter than argument 1

def test_f():
  issue_no = '18004'
  print('Jax issue no.', issue_no)
  jax.print_environment_info()

  with pytest.raises(ValueError) as e_info:
    f()
  print(f'{e_info.type.__name__}: {e_info.value}')
