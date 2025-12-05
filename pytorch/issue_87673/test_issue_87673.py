import pytest
import torch

def test_f():

    input_tensor = torch.rand(10, 10)
    grad_tensor = torch.rand(10, 10)
    index_tensor = torch.randint(low=0, high=10, size=(10,))

    with pytest.raises(RuntimeError) as e_info:
        input_tensor.addcdiv(1, grad_tensor, index_tensor)
        print(input_tensor)
    print(f'{e_info.type.__name__}: {e_info.value}')