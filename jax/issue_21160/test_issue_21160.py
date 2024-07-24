import functools

import jax
from jax.core import Primitive, ShapedArray, abstract_token
from jax.interpreters import ad, xla, mlir

from jax._src.effects import (
    Effect,
    lowerable_effects,
    ordered_effects,
)

import pytest

# Register an ordered effect
class EffectType(Effect):
    pass

lowerable_effects.add_type(EffectType)
ordered_effects.add_type(EffectType)

ordered_effect = EffectType()

# Define custom Jax primitive
custom_p = Primitive("custom")
custom_impl = functools.partial(xla.apply_primitive, custom_p)


def custom(x, token):
    return custom_p.bind(x, token)


def custom_xla_encode_cpu(ctx, x, token):
    effect = ctx.tokens_in.get(ordered_effect)[0]
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (effect,)}))
    return x, token


def custom_abstract_eval(xs, _):
    return (ShapedArray(xs.shape, xs.dtype), abstract_token), {ordered_effect}


custom_p.def_impl(custom_impl)
custom_p.def_effectful_abstract_eval(custom_abstract_eval)
custom_p.multiple_results = True

mlir.register_lowering(custom_p, custom_xla_encode_cpu, platform="cpu")


def test_f():
    issue_no = '21160'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    x = jax.numpy.zeros(100)
    token = jax.lax.create_token(None)
    devices = jax.local_devices()
    device = devices[0]
    # all good without this line
    with pytest.raises(AttributeError) as e_info:
        x = jax.device_put(x, device=device)
        custom(x, token) # AttributeError: 'AbstractToken' object has no attribute 'ndim'
    print(f'{e_info.type.__name__}: {e_info.value}')
