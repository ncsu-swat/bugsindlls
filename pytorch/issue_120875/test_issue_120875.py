import torch
import pytest


def f():

    input_data = torch.tensor([1+2j, 3-4j, 5j, 6])
    result_cpu = torch.all(input_data)

    input_data_cuda = input_data.cuda()
    result_cuda = torch.all(input_data_cuda).cpu()

    return result_cpu, result_cuda


def test_f():

    result_cpu, result_cuda = f()

    # They should be eqaul
    assert result_cpu != result_cuda

