import torch
import pytest
import os
from torch._dynamo.exc import BackendCompilerFailed

@torch.compile()
def call(data):
	return data.view(torch.uint8) + 1

def test_f():
    os.system('python -m torch.utils.collect_env')
    data = torch.randint(0, 2**4, [4096, 4096], device='cuda', dtype=torch.uint8)
    out = call(data) #OK
    with pytest.raises(BackendCompilerFailed) as e_info:
        out = call(data.view(torch.float16)) #Error 
    print(f'{e_info.type.__name__}: {e_info.value}')