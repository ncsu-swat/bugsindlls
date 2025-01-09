import torch
import pytest

# using fullgraph to raise an exception in place of graph breaks
@torch.compile(backend='inductor', fullgraph=True) 
def f(x):
    if hasattr(x, "attr"):
        return x + 1
    else:
        return x - 1

def test_f():
    issue_no = '129742'
    print('Pytorch issue no.', issue_no)

    t1 = torch.tensor([6.])
    t1.attr = False
    with pytest.raises(torch._dynamo.exc.Unsupported) as e_info:
        f(t1)
    print(f'{e_info.type.__name__}: {e_info.value}')
