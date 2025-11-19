import torch
import pytest

def test_f():
    
    torch.use_deterministic_algorithms(True)
    input = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    index = torch.tensor([0, 1, 1, 1, 2])
    src = torch.tensor([11, 12, 13, 14, 15])
    resultCpu = torch.Tensor.scatter(input, 0, index, src)
    print(resultCpu)

    inputCuda = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()
    indexCuda = torch.tensor([0, 1, 1, 1, 2]).cuda()
    srcCuda = torch.tensor([11, 12, 13, 14, 15]).cuda()
    resultCuda = torch.Tensor.scatter(inputCuda, 0, indexCuda, srcCuda)
    print(resultCuda)
    
    assert not torch.equal(resultCpu, resultCuda.cpu())