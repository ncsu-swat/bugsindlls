import torch
import pytest
import numpy as np
import torch.nn as nn
import traceback

def test_f():
    def forward(x, device):
        x = torch.var(out=x, correction=4,dim=0,input=torch.rand([], dtype=torch.float32).to('cpu'),keepdim=True)        
        return x

    input_tensor = torch.rand([10, 9, 8], dtype=torch.float32).to('cpu')
    cuda_tensor = input_tensor.clone().to('cuda')
    no_op_info = forward(input_tensor, 'cpu')
    print("build succeded")

    with pytest.raises(RuntimeError) as e_info:
        op_info = torch.compile(forward, mode='reduce-overhead',fullgraph=True,dynamic=True)(cuda_tensor, 'cuda')
        print(op_info)
    print(f'{e_info.type.__name__}: {e_info.value}')