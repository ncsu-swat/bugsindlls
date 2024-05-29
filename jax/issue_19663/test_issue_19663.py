import functools
import pytest
import jax
from jax import numpy as jnp
from jax import sharding
from jax.experimental import mesh_utils
from jax.experimental import shard_map

mesh = sharding.Mesh(
    mesh_utils.create_device_mesh((1, 1), jax.devices()[:1]), ('x', 'y')
)


@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(sharding.PartitionSpec(('x', 'y')),),
    out_specs=sharding.PartitionSpec(('x', 'y')),
)
def shmap(x):
  return jax.lax.all_to_all(
      x, ('x', 'y'), split_axis=1, concat_axis=1, tiled=True
  )

def test_f():
    issue_no = '19663'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    with pytest.raises(NotImplementedError) as e_info:
        shmap(jnp.arange(64).reshape((8, 8)))
    print(f'{e_info.type.__name__}: {e_info.value}')
