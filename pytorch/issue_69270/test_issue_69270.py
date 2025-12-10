import pytest
import torch
import sys

def test_f():
    tensor = torch.rand(torch.Size([2, 2, 4]), dtype=torch.float32)
    sections = 0 

    with pytest.raises(RuntimeError) as excinfo:
        torch.hsplit(tensor, sections)
    
    assert "cannot be split into 0 sections" in str(excinfo.value) or "sections must be > 0" in str(excinfo.value)
