import torch
import pytest

def test_f():
    # Simple test case
    tensor_data = [
        [1.0, float('nan')],
        [-1.0, 0.0]
    ]

    # Test on CPU - raises exception
    with pytest.raises(RuntimeError) as e_info:
        cpu_tensor = torch.tensor(tensor_data, dtype=torch.float16)
        result = torch.clamp_min(cpu_tensor, 4.35294e+26)
        print("CPU result:", result)
    print(f'{e_info.type.__name__}: {e_info.value}')

    # Test on CUDA - silently succeeds with inf values
    if torch.cuda.is_available():
        cuda_tensor = torch.tensor(tensor_data, device='cuda', dtype=torch.float16)
        result = torch.clamp_min(cuda_tensor, 4.35294e+26)
        print("CUDA result:", result)
        assert True
