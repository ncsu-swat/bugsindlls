import torch
import pytest
import torch.nn as nn

def test_f():

    def forward(x, y):
        return torch.nn.functional.hinge_embedding_loss(input=x, target=y, margin=0.11, reduction='sum')        

    x = torch.tensor([-668044.5], dtype=torch.float32)
    y = torch.rand([1], dtype=torch.float32)

    # Run in eager mode
    no_op_info = forward(x, y)
    print("Eager result:", no_op_info)
    
    # Run in compiled mode
    op_info = torch.compile(forward, mode='reduce-overhead', fullgraph=True)(x, y)
    print("Compiled result:", op_info)

    # Check if both results are close
    assert not torch.allclose(no_op_info.to('cpu'), op_info.to('cpu'), rtol=1e-3, atol=1e-3, equal_nan=True)
