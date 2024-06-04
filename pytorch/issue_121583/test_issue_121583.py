import torch
import torch.backends
import pytest
import os
import sys

def test_f():
    # Check whether cuurent os is MacOS with MPS supported
    if sys.platform != 'darwin' or not torch.backends.mps.is_available():
        pytest.skip('This test is only for MacOS with MPS enabled')
        exit(2)

    os.system('python -m torch.utils.collect_env')
    sdxl_latent_rgb_factors = torch.tensor(
        [
            #   R        G        B
            [0.3816, 0.4930, 0.5320],
            [-0.3753, 0.1631, 0.1739],
            [0.1770, 0.3588, -0.2048],
            [-0.4350, -0.2644, -0.4289],
        ],
        dtype=torch.bfloat16,
        device='mps',
    )
    samples2 = torch.zeros(182, 182, 4, 
        dtype=torch.bfloat16,
        device='mps',
    )
    with pytest.raises(RuntimeError) as e_info:
        x = torch.matmul(samples2, sdxl_latent_rgb_factors)
    print(f'{e_info.type.__name__}: {e_info.value}')
