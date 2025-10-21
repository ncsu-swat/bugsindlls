import torch
from torch import nn
import pytest

def test_f():

    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()

        def forward(self, inputs):
            return torch.cosh(**inputs)

    ip_size = [1]
    input_tensor = torch.randn(ip_size)
    out_tensor = torch.empty([2])
    cuda_inputs = input_tensor.clone().to('cuda')
    cuda_out = out_tensor.clone().to('cuda')

    mymodel = CustomModel()
    no_op_info= mymodel({'input': input_tensor, 'out': out_tensor})
    print("no_op_info", no_op_info)
    mymodel.to('cuda')
    
    with pytest.raises(AssertionError) as e_info: 
        op_info = torch.compile(mymodel)({'input': cuda_inputs, 'out': cuda_out})
        print(op_info)
    print(f'{e_info.type.__name__}: {e_info.value}')