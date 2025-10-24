import torch
import pytest
import numpy as np
import torch.nn as nn
import traceback

def test_f():
    
    def forward(x, device):
        x = torch.nn.functional.adaptive_max_pool3d_with_indices(output_size=x, input=torch.rand([9, 10, 9, 8, 6], dtype=torch.float32),return_indices=True)        
        return x

    inf = float('inf')
    nan = float('nan')
    is_valid = True
    input_tensor = 5
    cuda_tensor = input_tensor
    no_op_info = forward(input_tensor, 'cpu')
    print("build succeded")

    with pytest.raises(RuntimeError) as e_info:
        op_info = torch.compile(forward, mode='max-autotune',fullgraph=False,dynamic=True)(cuda_tensor, 'cuda')
    print(f'{e_info.type.__name__}: {e_info.value}')