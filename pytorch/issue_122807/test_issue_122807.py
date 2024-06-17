import os
import torch
import torch.distributed as dist
from datetime import timedelta

# Set environment variables
os.environ['RANK'] = '0'  # Set this to the appropriate rank for each process
os.environ['WORLD_SIZE'] = '1'  # Total number of processes
os.environ['MASTER_ADDR'] = 'localhost'  # Master node address
os.environ['MASTER_PORT'] = '12355'  # Master node port


def f():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=10))
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Create tensors
    s = 100 * 1024 * 1024
    with dist._coalescing_manager(device=device, async_ops=True) as cm:
        dist.all_reduce(torch.ones(s, device=device))
        dist.all_reduce(torch.ones(s, device=device))

    return len(cm.works)


def test_f():
    issue_no = '122842'
    print('Pytorch issue no.', issue_no)

    get_cm_len = f()

    # should be 1 rather than 0, and the latter is a no-op, which causes red-before-write issues whenever _coalescing_manager is used.
    assert get_cm_len == 0 
