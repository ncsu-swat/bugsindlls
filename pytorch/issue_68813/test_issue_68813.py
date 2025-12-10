import pytest
import torch
import sys

VALID_MAX_INDEX = 4 
INVALID_K = 6 

def test_f():
    arg_1 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)

    res1 = torch.kthvalue(arg_1, INVALID_K)
    res2 = arg_1.kthvalue(INVALID_K)
    res3 = torch.kthvalue(arg_1, INVALID_K)
    
    returned_index = res1.indices.item()
    
    assert returned_index > VALID_MAX_INDEX, (
        f"BUG NOT REPRODUCED: The returned index "
        f"({returned_index}) was within the valid range (0-{VALID_MAX_INDEX}). "
    )