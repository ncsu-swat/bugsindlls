import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_alike import shard_alike

def test_f():
    issue_no = '19459'
    print('Jax issue no.', issue_no)
    devices = jax.devices()
    print(devices)
    # [CpuDevice(id=0)]

    mesh = Mesh(devices, axis_names=("i"))
    sharding = NamedSharding(mesh, P('i', None))

    x = jax.device_put(jnp.zeros((len(devices), 10)), sharding)
    x_sharding = x.sharding
    # NamedSharding(mesh=Mesh('i': 1), spec=PartitionSpec('i', None))

    _, y = shard_alike(x, jnp.ones(x.shape))
    y_sharding = y.sharding
    # SingleDeviceSharding(device=CpuDevice(id=0))

    assert x_sharding != y_sharding
    # Expected behavior is they'd be equal
