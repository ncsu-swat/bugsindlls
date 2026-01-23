import pytest
import torch
import torch.nn as nn

def test_f():
    padding = [-1, -2, 1, 1]
    c1 = torch.nn.ReflectionPad2d(padding)
    c2 = torch.nn.ReplicationPad2d(padding)
    c3 = torch.nn.ConstantPad2d(padding, 0)
    c4 = torch.nn.ZeroPad2d(padding)
    input = torch.rand([1, 1, 3, 3], dtype=torch.float32)

    print(c1(input))
    print(c2(input))

    with pytest.raises(RuntimeError) as e_info:
        c3(input)
        # RuntimeError: The input size 3, plus negative padding -1 and -2 resulted in a negative output size, which is invalid. Check dimension 3 of your input.
        c4(input)
        # RuntimeError: The input size 3, plus negative padding -1 and -2 resulted in a negative output size, which is invalid. Check dimension 3 of your input.
    print(f"{e_info.type.__name__}: {e_info.value}")