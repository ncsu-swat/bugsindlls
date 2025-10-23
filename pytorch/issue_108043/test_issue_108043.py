import torch
from torch import nn
import pytest
import numpy as np

def test_f():
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.layer1 = nn.ConstantPad2d(padding=0, value=1)
            
        def forward(self, inputs):
            return self.layer1(inputs)

    ip_size = [0, 0, 1]
    input_tensor = torch.randn(ip_size)
    cuda_inputs = input_tensor.clone().to('cuda')

    mymodel = CustomModel()
    no_op_info = mymodel(input_tensor)
    mymodel.to('cuda')
    
    with pytest.raises(RuntimeError) as e_info:
        op_info = torch.compile(mymodel.forward, mode='max-autotune')(cuda_inputs)
        print(op_info)
    print(f'{e_info.type.__name__}: {e_info.value}')