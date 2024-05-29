import jax
import jax.numpy as jnp
import jax.nn as nn
import pytest
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, Mesh, PartitionSpec
from flax import linen as nn


class Model(nn.Module):
    @nn.compact
    def __call__(self, x, add_noise):
        x = nn.Dense(1)(x)
        return jnp.where(
            add_noise, x + jax.random.normal(self.make_rng('params'), x.shape), x
        )


def f():

    device_mesh = mesh_utils.create_device_mesh((1,))
    mesh = Mesh(devices=device_mesh, axis_names=('data',))
    data_sharding = NamedSharding(mesh, PartitionSpec('data',))

    module = Model()
    init_rng, apply_rng = jax.random.split(jax.random.key(0))
    # get 8 different rng's that will be used by the 8 devices when doing forward inference
    apply_rng = jax.random.split(apply_rng, 1)
    x = jnp.ones((1, 1))
    variables = module.init(init_rng, x, False)

    def forward(variables, x, add_noise, rng_key_batch):
        # rng_key_batch is a batch of size 1 containing 1 PRNG key
        # index slice into the rng_key_batch to access the PRNG key
        return module.apply(
            variables, x, add_noise, rngs={'params': rng_key_batch[0]}
        )

    # define partition specifications
    data_pspec = PartitionSpec('data')
    no_pspec = PartitionSpec()

    shmap_forward = shard_map(
        forward,
        mesh=mesh,
        in_specs=(no_pspec, data_pspec, no_pspec, data_pspec),
        out_specs=data_pspec,
    )

    out = shmap_forward(variables, x, True, apply_rng)

def test_f():
    issue_no = '18851'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    with pytest.raises(NotImplementedError) as e_info:
        f()
    
    print(f'{e_info.type.__name__}: {e_info.value}')
