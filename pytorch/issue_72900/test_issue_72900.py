import pytest
import torch
import torch.nn as nn

def test_f():

    results = dict()
    row = 0
    col = 1
    offset = 2
    
    results["res_1"] = torch.tril_indices(row, col, offset=offset)
    print(results["res_1"])
    assert results["res_1"] is not None
    
    with pytest.raises(RuntimeError) as e_info:
        results["res_2"] = torch.triu_indices(row,col,offset=offset)
        print(results["res_2"])
    print(f"{e_info.type.__name__}: {e_info.value}")