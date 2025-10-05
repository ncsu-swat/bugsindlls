
import torch
import pytest
import numpy as np

def arrays_bitwise_equal(a, b):
    # Checks both value and sign bit
    return np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))

def test_f():
    torch.manual_seed(420)
    torch.cuda.manual_seed_all(420)
    x = torch.tensor([-0., 0., 0.])
    cpu_clamp = torch.clamp(x, min=0, max=1).numpy()  # prints [-0., 0., 0.]
    print(cpu_clamp)

    x = x.cuda()
    gpu_clamp = torch.clamp(x, min=0, max=1).cpu().numpy()  # prints [0., 0., 0.]
    print(gpu_clamp)

    x = np.array([-0., 0., 0.])
    numpy_clip = np.clip(x, 0, 1)
    print(numpy_clip)  # prints [0., 0., 0.]

    # Check mismatch in histc results
    assert arrays_bitwise_equal(numpy_clip, gpu_clamp)
    assert not arrays_bitwise_equal(gpu_clamp, cpu_clamp)
    assert not arrays_bitwise_equal(numpy_clip, cpu_clamp)
