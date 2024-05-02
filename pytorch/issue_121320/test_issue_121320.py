import torch

issue_no = '121320'
print('Pytorch issue no.', issue_no)
torch.utils.collect_env.main()

FIFTY_MIL_CYCLES = 50000000

def test_f():
    a=torch.ones(1, device="cuda")*10
    b=torch.empty(1, device="cpu", pin_memory=True)
    c=torch.empty(1, device="cuda", dtype=torch.long)

    torch.cuda._sleep(FIFTY_MIL_CYCLES)
    b.copy_(a, non_blocking=True)
    c.copy_(b, non_blocking=True) # c is 0, not 10
    # without type conversion it works
    c_float = torch.empty(1, device="cuda", dtype=torch.float)
    b.fill_(0)
    torch.cuda._sleep(FIFTY_MIL_CYCLES)
    b.copy_(a, non_blocking=True)
    c_float.copy_(b, non_blocking=True) # it is 10
    assert c != c_float
    assert c == 0
