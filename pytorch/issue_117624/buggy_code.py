import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, out, input, other):
        return torch.logical_or(out=out, input=input, other=other)        

x = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float16)
y = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float32)
z = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float32)

model = Model().to(torch.device('cpu'))
eag = model(x, y, z)
opt = torch.compile(model.forward, mode='max-autotune')(x, y, z)


same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
        raise ValueError('diff value')