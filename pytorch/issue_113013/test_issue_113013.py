import torch

def test_f():
    def forward(x):      
        x = torch.adaptive_avg_pool1d(input=x, output_size=2)
        x = torch.argmax(input=x)
        return x

    x = torch.rand([3, 3, 3], dtype=torch.float64)

    # Eager execution
    eager = forward(x)
    print("Eager result:", eager)

    # Compiled execution
    comp = torch.compile(
        forward, mode='max-autotune-no-cudagraphs', fullgraph=True, dynamic=True
    )(x)
    print("Compiled result:", comp)
        
    assert not torch.allclose(eager.to('cpu'), comp.to('cpu'), rtol=1e-3, atol=1e-3, equal_nan=True)