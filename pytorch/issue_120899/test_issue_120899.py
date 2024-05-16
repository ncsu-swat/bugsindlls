import torch
import sys

def test_f():
    # Check whether cuurent os is MacOS with MPS supported
    if sys.platform != 'darwin' or not torch.backends.mps.is_available():
        pytest.skip('This test is only for MacOS with MPS enabled')
        exit(2)

    issue_no = '120899'
    print('Pytorch issue no.', issue_no)

    t_mps = torch.tensor([torch.nan, 1, 2], device="mps")
    a_mps = torch.clamp(t_mps, min=-100, max=100)
    # tensor([-100.,    1.,    2.], device='mps:0')
    b_mps = torch.clamp(t_mps, min=-100)
    # tensor([-100.,    1.,    2.], device='mps:0')
    c_mps = torch.clamp(t_mps, max=100)
    # tensor([100.,   1.,   2.], device='mps:0')

    t_cpu = torch.tensor([torch.nan, 1, 2], device="cpu")
    a_cpu = torch.clamp(t_cpu, min=-100, max=100)
    # tensor([nan, 1., 2.])
    b_cpu = torch.clamp(t_cpu, min=-100)
    # tensor([nan, 1., 2.])
    c_cpu = torch.clamp(t_cpu, max=100)
    # tensor([nan, 1., 2.])

    print(a_mps.to('cpu').numpy())
    print(a_cpu.to('cpu').numpy())
    # inconsistent behavior
    assert (a_mps.to('cpu').numpy() != a_cpu.to('cpu').numpy()).any()
    assert (b_mps.to('cpu').numpy() != b_cpu.to('cpu').numpy()).any()
    assert (c_mps.to('cpu').numpy() != c_cpu.to('cpu').numpy()).any()
