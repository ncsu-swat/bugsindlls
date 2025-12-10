import pytest
import torch
import torch.nn as nn

def test_f():
    # 72900
    row = 0
    col = 1
    offset = 2
    
    result = torch.tril_indices(row, col, offset=offset)
    
    bug_reproduced = result.numel() > 0
    
    assert bug_reproduced