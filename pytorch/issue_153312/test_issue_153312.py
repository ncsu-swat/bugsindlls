import pytest
import torch

def test_f():

    print(torch.__version__)
    cpu_result=torch.tensor([[0.],
        [0.],
        [0.]], dtype=torch.float64)
    tensor = torch.tensor([[ -7.8130e-88, -2.2092e-138,  -1.8673e+03, -7.6272e-253,  3.9203e+110,
            1.8380e-51,  2.8762e+268,  2.9094e+286,  5.1816e-228, -4.4916e+191,
            -7.4057e+80,  -9.1955e-18,  5.6536e+225,  8.8364e-175,  1.5053e-226],
            [-3.0521e+239, -2.8307e+306,   1.3297e-03, -9.9969e-132,  2.8920e-286,
            2.3964e+58, -6.8138e-281,  2.0321e-305,  -3.5127e+74,  -4.7560e-92,
            -8.9403e-99, -1.9739e-187, -2.5124e-173,  2.0458e+295,   4.4992e+52],
            [  6.8752e+21,  1.9332e+189, -8.6940e-189,  -6.6743e-15,   1.4691e+41,
            1.0338e+63,  -2.0779e-28, -7.6642e+104,  1.3390e+284, -8.0859e+194,
            8.4600e+107,   4.9115e-44,  1.1665e+285,  5.1275e+203,  9.7580e+303]],
        dtype=torch.float64)

    try:
        res = torch.nn.functional.lp_pool1d(
            tensor,
            norm_type=-1.38119e+150,
            kernel_size=7879455037536781369,
            ceil_mode=True,
        )
        print("CPU result:", res)
        assert torch.allclose(res, cpu_result)
    except RuntimeError as e:
        print(f"CPU error: {e}")

    tensor_gpu = tensor.to("cuda:0")
    with pytest.raises(RuntimeError) as e_info:
        res = torch.nn.functional.lp_pool1d(
            tensor_gpu,
            norm_type=-1.38119e+150,
            kernel_size=7879455037536781369,
            ceil_mode=True,
        )
        print("GPU result:", res)
    print(f'{e_info.type.__name__}: {e_info.value}')
