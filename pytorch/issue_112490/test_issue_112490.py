import torch
import torch.nn as nn
import pytest
import traceback

def test_f():

    args =  {'input': torch.rand([]), 
             'out': torch.rand([2, 2, 8, 2, 6, 6, 8, 1, 7], dtype=torch.float64), 
             'tensor1': torch.rand([6]), 
             'tensor2': torch.rand([1]), 
             'value': 2}
    with pytest.raises(RuntimeError) as e_info:
        res = torch.addcdiv(**args)
        print(res)
    print(f'{e_info.type.__name__}: {e_info.value}')