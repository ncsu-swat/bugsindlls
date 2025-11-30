import torch
import pytest

def test_f():

    # Input bfloat16 tensor
    tensor_bf16 = torch.tensor([
        [ 1.0986e-25, -1.9114e+30, -9.5291e-20, -4.3163e-39, -1.2337e+23,
        2.0698e+30,  1.1934e+30],
        [-4.3910e-18, -4.3910e-18, -4.3910e-18, -4.3910e-18,  1.1934e+30,
        1.1934e+30, -4.4256e+28],
        [-4.9304e-29, -3.2969e-11, -8.0691e+05, -8.8500e+08, -2.0938e+00,
        -2.4629e+16,  5.6820e+26]], dtype=torch.bfloat16)

    # Cumsum with dtype=torch.int8 on CPU
    cpu_res = torch.cumsum(tensor_bf16, -1, dtype=torch.int8)

    # Cumsum with dtype=torch.int8 on GPU
    # Ensure CUDA is available before running this part
    if torch.cuda.is_available():
        gpu_res = torch.cumsum(tensor_bf16.to("cuda"), -1, dtype=torch.int8)
        gpu_res_cpu = gpu_res.to("cpu")
        difference = cpu_res - gpu_res_cpu
    else:
        gpu_res = "CUDA not available"
        gpu_res_cpu = "CUDA not available"
        difference = "CUDA not available"

    print("Input bfloat16 tensor:")
    print(tensor_bf16)

    print("\nCPU cumsum result (dtype=torch.int8):")
    print(cpu_res)

    print("\nGPU cumsum result (dtype=torch.int8) (moved to CPU for printing):")
    print(gpu_res_cpu)

    assert not torch.equal(cpu_res, gpu_res_cpu), "CPU and GPU results match!"
    
    print("\nDifference (CPU - GPU):")
    print(difference)

    # For calculating expected behavior:
    # 1. Explicitly cast bfloat16 tensor to int8
    tensor_int8_explicit = tensor_bf16.to(torch.int8)
    # 2. Perform cumsum on the explicitly casted int8 tensor
    expected_res = torch.cumsum(tensor_int8_explicit, -1, dtype=torch.int8) # dtype ensures int8 accumulation

    print("\n--- Expected Behavior Analysis ---")
    print("\nInput tensor explicitly cast to int8:")
    print(tensor_int8_explicit)
    print("\nExpected cumsum result (after explicit cast to int8):")
    print(expected_res)
