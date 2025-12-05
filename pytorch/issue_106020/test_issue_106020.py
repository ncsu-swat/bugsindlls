import torch
import pytest
from torch import nn

def test_f():

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()

        def forward(self, x, y):
            loss = nn.MultiMarginLoss(weight=torch.tensor([[2],[3]]))
            return loss(x, y)
        
    model = MyModel()
    x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
    y = torch.tensor([3]) 
    with pytest.raises(AssertionError) as e_info:
        model(x, y)
    print(f'{e_info.type.__name__}: {e_info.value}')