import pytest
import torch
import sys

def test_f():
    tensor_0 = torch.rand(torch.Size([]))
    tensor_1 = torch.rand(torch.Size([3]))
    tensors = [tensor_0, tensor_1]

    result = torch.hstack(tensors)

    assert result.dim() == 1 and result.size(0) == 4, (
        f"Unexpected result: {result.shape}"
    )
