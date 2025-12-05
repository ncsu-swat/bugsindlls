import torch
import pytest

def test_f():

    print("Torch version:", torch.__version__)

    shape = (6, 6, 6, 7)
    tensor = torch.zeros(shape, dtype=torch.int32)
    tensor[0, 0, 0, 1] = -1929248768
    tensor[0, 0, 0, 2] = -151113212
    tensor[0, 0, 1, 0] = 1241368915
    tensor[0, 0, 2, 0] = -1240897712
    tensor[0, 1, 0, 0] = -33686019
    tensor[0, 0, 3, 3] = 839159266

    try:
        cpu_result = torch.aminmax(tensor, keepdim=True)
        print("CPU result:", cpu_result)
        print("CPU shape (min):", cpu_result.min.shape)
    except Exception as e:
        print("CPU error:", e)

    tensor_gpu = tensor.to("cuda")
    try:
        gpu_result = torch.aminmax(tensor_gpu, keepdim=True)
        print("GPU result:", gpu_result)
        print("GPU shape (min):", gpu_result.min.shape)
    except Exception as e:
        print("GPU error:", e)
    
    assert cpu_result.min.shape != gpu_result.min.shape
