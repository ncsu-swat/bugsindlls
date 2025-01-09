import torch
import pytest

class FunctionalConv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

    def forward(self, x, weight, bias):
        return torch.nn.functional.conv2d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FunctionalConv2d()
        self.conv2 = FunctionalConv2d()

    def forward(self, x, weight, bias):
        x = self.conv1(x, weight, bias)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, weight, bias)
        return x
    
def test_f():
    issue_no = '117033'
    print('Pytorch issue no.', issue_no)

    dtype = torch.float64  # Disable XNNPACK
    inputs = (torch.randn(1, 3, 5, 5, dtype=dtype), torch.randn(3, 3, 3, 3, dtype=dtype), torch.rand(3, dtype=dtype))
    with pytest.raises(Exception) as e_info:
        with torch.backends.mkldnn.flags(False): # Disable MKLDNN
            gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)

    print(f'{e_info.type.__name__}: {e_info.value}')


