import pytest
import torch
import torch.nn.functional as F

def test_f():

    input = torch.randn(1, 3, 5, 5)
    weight = torch.randn(1, 3, 1, 1)

    with pytest.raises(RuntimeError) as e_info:
        output = F.prelu(input, weight)
        print(output)
    print(f'{e_info.type.__name__}: {e_info.value}')