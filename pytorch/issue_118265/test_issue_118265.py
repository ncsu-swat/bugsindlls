import torch
import torch.nn as nn

def test_f():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, out, input, other):
            out = torch.bitwise_right_shift(out=out, input=input, other=other)        
            return out

    out1 = torch.randint(-32768, 32767, [7, 2, 6, 7], dtype=torch.int16)
    out2 = out1.clone()
    input = torch.randint(0, 100, [7, 2, 6, 7], dtype=torch.int64)
    other = torch.randint(0, 100, [7, 2, 6, 7], dtype=torch.int64)

    model = Model().to(torch.device('cpu'))
    eag = model(out1, input, other)
    opt = torch.compile(model.forward, mode='reduce-overhead')(out2, input, other)

    # Check if both results are close
    assert not torch.allclose(eag.to('cpu'), opt.to('cpu'), rtol=1e-3, atol=1e-3, equal_nan=True)