import torch
import torch.nn as nn

def test_f():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input): 
            input = torch.diag_embed(input=input, dim1=-1,dim2=0,offset=1)        
            return input

    x = torch.rand([8, 6, 8, 6, 6, 1], dtype=torch.float64)
    model = Model().to(torch.device('cpu'))
    eag = model(x)

    opt = torch.compile(model.forward)(x)

    # Check if both results are close
    assert not torch.allclose(eag.to('cpu'), opt.to('cpu'), rtol=1e-3, atol=1e-3, equal_nan=True)