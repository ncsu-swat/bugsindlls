import torch
import pytest

def test_f():

    groups = 2
    m = torch.nn.ChannelShuffle(groups=groups)
    input_tensor = torch.rand([6, 0, 6], dtype=torch.float32).to('cuda')
    m.to('cuda')
    with pytest.raises(NotImplementedError) as e_info: 
        m(input_tensor)
    print(f'{e_info.type.__name__}: {e_info.value}')
