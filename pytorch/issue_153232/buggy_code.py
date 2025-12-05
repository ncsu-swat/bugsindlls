import torch

# Create a bfloat16 tensor with extreme values
lu_data = torch.tensor([[[-8.2838e+07,  5.8933e-32, -3.4757e+25,  1.1114e+21,  1.6466e+06,
          -9.5569e+29,  1.1391e+21],
         [ 3.3389e+21,  1.5088e-21, -1.3733e-10, -4.9472e-06,  3.8554e+21,
           1.0156e+00, -6.2604e-26],
         [-2.6258e-19,  9.3174e-20,  1.4809e+25, -8.0220e+15, -1.1482e-11,
           3.0264e+19,  3.3499e+22],
         [-1.0695e+09, -1.9173e+31, -1.6516e-09, -1.8507e-34,  5.5320e+27,
          -5.8620e-13,  1.1511e+22]],
], dtype=torch.bfloat16)

# Empty pivots tensor
lu_pivots = torch.tensor([], dtype=torch.int32)

# Call lu_unpack which triggers the segmentation fault
torch.lu_unpack(lu_data, lu_pivots, unpack_data=True, unpack_pivots=True)