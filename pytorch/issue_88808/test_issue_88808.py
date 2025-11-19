import pytest
import torch
import numpy as np

def test_f():
    
    input_data = torch.Tensor(np.random.uniform(size=(2, 3))).cuda()
    output = torch.tensor([[[[[[[[[[[[[[[[[[[[[[[[2.3200, 4.1619, 4.2059],
                                [3.3720, 2.7839, 3.6389]]]]]]]]]]]]]]]]]]]]]]]],device='cuda:0')
    input = torch.tensor([[[[[[[[[[[[[[[[[[[[[[[[[   float('nan'),    float('nan'),    float('nan')],
                                [   float('nan'),    float('nan'), 0.6918]]]]]]]]]]]]]]]]]]]]]]]]],device='cuda:0')
    output = torch.mean(torch.abs((output - input)), dim=(- 1), keepdim=True)
    output = torch.mean(torch.abs((output - input_data)), dim=(- 1), keepdim=True)
    output = torch.mean(torch.abs((output - torch.min(output))), dim=(- 1), keepdim=True)
    with pytest.raises(RuntimeError) as e_info:
        output = torch.median(output, dim=(- 1))
    print(f'{e_info.type.__name__}: {e_info.value}')
