import pytest
import torch

def test_f():

    input = torch.rand([1, 1, 2, 2], dtype=torch.float32)
    indices = torch.randint(-16,1024,[1, 1, 2, 2], dtype=torch.int64)
    kernel_size = [16, -1024]
    stride = [-16, 1]
    res=torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride)
    print(res)
    assert list(res.size()) == [1, 1, 0, -1023]
    # tensor([], size=(1, 1, 0, -1023))