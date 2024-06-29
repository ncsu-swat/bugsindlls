import os
from functools import partial
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

import jax
import jaxlib
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.tree_util import tree_map
import pytest

def make_custom_partitioning(part_func):
    @custom_partitioning
    def my_func(x):
        return jnp.sum(x)

    def partition(mesh, arg_shapes, result_shape):
        result_shardings = tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = tree_map(lambda x: x.sharding, arg_shapes)
        assert isinstance(arg_shardings[0], NamedSharding)
        assert (None, 'x') == arg_shardings[0].spec
        return mesh, partial(part_func, axis_name='x'), result_shardings, arg_shardings

    def infer_sharding(mesh, arg_shapes, result_shape):
        return NamedSharding(mesh, P())

    def propagate_user_sharding(mesh, user_shape):
        return user_shape.sharding

    my_func.def_partition(partition, infer_sharding, propagate_user_sharding=propagate_user_sharding)
    return my_func


def part_func(x, axis_name):
    def f(carry, part):
        carry += jax.lax.psum(jnp.sum(part), axis_name=axis_name)
        return carry, None
    return jax.lax.scan(f, 0, x)[0]
    


def test_f():
    issue_no = '20864'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    my_func = jax.jit(make_custom_partitioning(part_func))
    
    with Mesh(jax.devices(backend='cpu'), axis_names=('x',)) as mesh:
        array = jnp.ones([4, 4])
        assert int(my_func(array)) == 16
        array = jax.device_put(array, NamedSharding(mesh, P(None, 'x')))
        
        
        with pytest.raises(jaxlib.xla_extension.XlaRuntimeError) as e_info:
            assert int(my_func(array)) == 16

        print(f'{e_info.type.__name__}: {e_info.value}')


    




