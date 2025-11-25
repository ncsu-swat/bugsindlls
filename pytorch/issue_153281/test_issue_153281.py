import torch
import pytest

def test_f():
    input = torch.tensor([-1.0])
    lower = 0
    upper = float('inf')

    training = False

    out_gpu = torch.nn.functional.rrelu(input.cuda(), lower, upper, training=training).cpu()
    print(out_gpu) # tensor([-inf])

    out_cpu = torch.nn.functional.rrelu(input, lower, upper, training=training) # No error
    print(out_cpu) # tensor([-inf])
    
    assert torch.equal(out_cpu, out_gpu)
    
    training = True

    out_gpu = torch.nn.functional.rrelu(input.cuda(), lower, upper, training=training).cpu()
    print(out_gpu) # tensor([-inf])

    with pytest.raises(RuntimeError) as e_info:
        out_cpu = torch.nn.functional.rrelu(input, lower, upper, training=training) 
        # RuntimeError: Expected to - from <= std::numeric_limits<T>::max() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
        print(out_cpu)
    print(f'{e_info.type.__name__}: {e_info.value}')