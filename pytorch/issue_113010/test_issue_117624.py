import subprocess
import torch

def test_f():
    def forward(x, y):          
        return torch.abs(input=x, out=torch.t(input=y))

    x = torch.rand([9, 10], dtype=torch.float32)
    y = torch.rand([10, 9], dtype=torch.float32)

    # Run in eager mode
    eager = forward(x, y)
    print("Eager Result", eager)
    
    # Run in compiled mode
    compiled = torch.compile(forward, mode='default')(x, y)
    print("Compiled Result", compiled)
    
    # Check if both results are close
    assert not torch.allclose(eager.to('cpu'), compiled.to('cpu'), rtol=1e-3, atol=1e-3, equal_nan=True)