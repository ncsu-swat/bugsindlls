import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 73154
    input_tensor = torch.rand([1, 1, 2, 2], dtype=torch.float32)
    indices = torch.randint(-16, 1024, [1, 1, 2, 2], dtype=torch.int64)
    kernel_size = [16, -1024]
    stride = [-16, 1]
    
    result = F.max_unpool2d(input_tensor, indices, kernel_size, stride)

    bug_reproduced = any(dim < 0 for dim in result.shape)
    
    assert bug_reproduced