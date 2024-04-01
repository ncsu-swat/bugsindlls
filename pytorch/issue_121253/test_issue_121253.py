import os

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

import torch
import torch.utils.collect_env
import numpy as np

torch.utils.collect_env.main()

torch.manual_seed(42)

class Model(torch.nn.Module):

    def __init__(self, weight, bias):
        super(Model, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight, alpha=1.0, beta=0.1)

weight = 1
bias = 1
func = Model(torch.randn(64, 64), torch.randn(64)).to('cpu')

x = torch.randn(1, 64)

# x = x.to(dtype=torch.float64)
# func.weight = func.weight.to(dtype=torch.float64)
# func.bias = func.bias.to(dtype=torch.float64)

def test_f():
    with torch.no_grad():
        func1 = torch.compile(func)
        val_compiled = func1(x.clone())
        # tensor([[  5.5041,   8.0136,   9.7231,   4.3172,  -4.7227,   7.4632, -10.4915,
        #       -0.5536, -15.7986,   8.6184,   3.1239,  -1.8854,   7.8321,   3.1380,
        #        0.0261,   2.4693,  -0.1138,  -5.1792,  -4.9836,  -7.1234,   3.7069,
        #        6.9476,  22.1048,  -3.7576,  12.4651, -10.9513,   7.8899,   4.4365,
        #        0.0233,  12.4531,   5.4630,   2.7396,  15.0662, -11.8093,  -7.6297,
        #        1.5935,  -1.6402,  11.1157,  -2.1696, -17.6288,   4.0042,   2.1122,
        #      -16.2381,  19.1538,   0.2264,   0.5900,  -7.5213,  14.8976,  -6.1498,
        #      -11.5165,  -6.5250,   8.4473,   3.2056,   5.9807,  -8.0135,  -5.0815,
        #       -3.4325,  -2.4708,   3.6546,   1.3412,  -8.9707,  21.8540,   0.4926,
        #       -3.8654]])

        val_non_compiled = func(x.clone())
        # tensor([[  5.9718,   6.3465,   8.0702,   2.4505,  -4.0591,   8.1550, -10.4454,
        #       -1.9923, -15.9896,   7.6231,   1.9432,  -2.2672,   8.2752,   1.6473,
        #       -0.3439,   2.6879,  -0.8909,  -3.8750,  -5.2665,  -6.2186,   4.9161,
        #        5.8066,  23.2692,  -3.0903,  12.7620, -11.2484,   7.0066,   5.7786,
        #       -0.4614,  11.2506,   5.9704,   2.1425,  14.6779, -11.1137,  -6.9070,
        #        1.1754,  -1.4864,   8.6796,  -2.7658, -17.0402,   3.3492,   2.0289,
        #      -16.0763,  18.5234,   1.3519,  -0.2281,  -7.3793,  15.2927,  -6.8148,
        #      -11.0951,  -5.0884,   9.9478,   2.9006,   5.6432,  -6.8172,  -5.5889,
        #       -4.1571,  -3.1412,   3.8111,   0.3441,  -9.4339,  21.0498,   1.8529,
        #       -3.0991]])

        assert False == np.allclose(val_compiled, val_non_compiled, rtol = 1e-3, atol = 1e-3, equal_nan = False)