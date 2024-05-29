import torch
import pytest
import os

def channel_shuffle_fn(input):
  channel_shuffle = torch.nn.ChannelShuffle(2)
  return channel_shuffle(input)


def test_f():
  issue_no = '122771'
  print('Pytorch issue no.', issue_no)
  os.system('python -m torch.utils.collect_env')
    
  comp_model = torch.compile(channel_shuffle_fn, fullgraph=True)
  input = torch.randn(1, 4, 4, 2)
  output = comp_model(input)

  input = torch.randn(1, 4, 2, 2)


  with pytest.raises(torch._dynamo.exc.Unsupported) as e_info:  
    output = comp_model(input)

  print(f'{e_info.type.__name__}: {e_info.value}')