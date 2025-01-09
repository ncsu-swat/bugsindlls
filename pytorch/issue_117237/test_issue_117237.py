import torch
import pytest


def test_f():
    input1 = torch.randint(high=(1 << 7) - 1, size=[1], dtype=torch.int8)
    input2 = torch.randn(size=[1], dtype=torch.bfloat16)

    module = torch.nn.Bilinear(in1_features=1,in2_features=1,out_features=0,bias=True,dtype=torch.complex64)
    with pytest.raises(RuntimeError) as e_info:       
        output = module(input1, input2)
    print(f'{e_info.type.__name__}: {e_info.value}')


test_f()