import torch
import pytest
import os
import torch._higher_order_ops.map


def test_f():
    os.system('python -m torch.utils.collect_env')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # This should return 1 as expected
    assert torch.cuda.device_count() == 1

