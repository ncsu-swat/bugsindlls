import pytest
import torch

def test_f():
    normalized_shape = [1024]
    layer = torch.nn.LayerNorm(normalized_shape)
    input = torch.randint(0,8, [0, 1, 1024])
    with pytest.raises(RuntimeError) as e_info:
        res = layer(input)
    print(f"{e_info.type.__name__}: {e_info.value}")