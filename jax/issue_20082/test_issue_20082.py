import pytest

import jax

from jax.experimental.attrs import jax_getattr, jax_setattr


class StatefulRNG:
    key: jax.Array

    def __init__(self, key: jax.Array):
        self.key = key

    def split(self) -> jax.Array:
        key = jax_getattr(self, "key")
        new_key, returned_key = jax.random.split(key)
        jax_setattr(self, "key", new_key)
        return returned_key


rng = StatefulRNG(jax.random.key(0))


def jitted():
    rng.split()
    rng.split()


def test_f():
    jax.print_environment_info()

    with pytest.raises(AssertionError) as e_info:
        jax.jit(jitted)()

    print(e_info.value)
