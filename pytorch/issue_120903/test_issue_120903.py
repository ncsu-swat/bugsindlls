import torch
import numpy as np
import pytest

issue_no = 120903
seed = 120903

torch.manual_seed(seed)

def test_f():
    issue_no = '120903'
    print('Pytorch issue no.', issue_no)
    print('Seed: ', seed)

    input_data = torch.randn(3, 4, 5, 6)

    scale = np.array([0.1, 0.2, 0.3, 0.4])
    zero_point = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    axis = 1
    quant_min = 0
    quant_max = 255

    with pytest.raises(RuntimeError) as e_info:
        output = torch.fake_quantize_per_channel_affine(input_data, torch.from_numpy(scale), zero_point, axis, quant_min, quant_max)
    print(e_info.value)
