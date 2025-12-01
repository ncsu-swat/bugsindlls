import torch
import pytest

def test_f():

    print(torch.__version__)

    input_tensor = torch.tensor(-4015619983489386247)
    tensor1 = torch.tensor([
        [[ 25527, -21905, -12030,  13427,  -1546]],
    ], dtype=torch.int16)
    tensor2 = torch.tensor(-752003609, dtype=torch.int32)
    value_scalar = 988328483643648370

    # CPU version correctly raises error
    with pytest.raises(RuntimeError) as e_info:
        res = torch.addcmul(input_tensor, tensor1, tensor2, value=value_scalar)
        print("CPU result:", res)
    print(f'{e_info.type.__name__}: {e_info.value}')

    # CUDA version silently returns potentially incorrect results
    input_tensor_gpu = input_tensor.cuda()
    tensor1_gpu = tensor1.cuda()
    tensor2_gpu = tensor2.cuda()
    try:
        res = torch.addcmul(input_tensor_gpu, tensor1_gpu, tensor2_gpu, value=value_scalar)
        print("CUDA result:", res)
        assert True
    except RuntimeError as e:
        print("CUDA error:", e)

