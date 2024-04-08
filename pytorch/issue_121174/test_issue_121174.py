import torch
import math
import pytest

torch.manual_seed(0)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.q_conv = torch.nn.Conv2d(4, 4, 1)
        self.k_conv = torch.nn.Conv2d(4, 4, 1)
        self.v_conv = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x):
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        div = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weight = torch.nn.functional.softmax(div, dim=-1)
        output = torch.matmul(attn_weight, v)
        return output
def test_f():
    x = torch.randn(1, 4, 2, 2)

    func = Model()

    with torch.no_grad():
        # tensor([[[[ 0.1035, -0.2616],
        #         [ 0.0461, -0.3923],
        #         [ 0.0852, -0.4379],
        #         [ 0.0538, -0.5618]],
        #         [[-0.3332, -0.2985],
        #         [-0.1777, -0.2978],
        #         [-0.4235, -0.3401],
        #         [-0.4361, -0.3477]]]])

        func1 = torch.compile(func)
        # tensor([[[[ 0.1624,  0.2315],
        #         [ 0.0282,  0.0822],
        #         [ 0.2613,  0.0440],
        #         [-0.0161, -0.0560]],

        #         [[-0.3225, -0.2300],
        #         [-0.2753, -0.3049],
        #         [-0.3932, -0.3184],
        #         [-0.3870, -0.2880]]]])


        torch.testing.assert_close(func.k_conv.weight, func1.k_conv.weight) # True
        torch.testing.assert_close(func.q_conv.weight, func1.q_conv.weight) # True
        torch.testing.assert_close(func.v_conv.weight, func1.v_conv.weight) # True

        with pytest.raises(AssertionError) as e_info:
            torch.testing.assert_close(func(x.clone()), func1(x.clone()))
        print(e_info.value)