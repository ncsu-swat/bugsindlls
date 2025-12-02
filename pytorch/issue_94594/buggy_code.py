import torch
input = torch.rand([2, 3, 13, 15, 16, 1], dtype=torch.float32)
bins = []
range_0 = -11.114022798233492
range_1 = 8.42630999399097
range = [range_0, range_1, ]
weight = torch.rand([4, 10, 1, 1, 15, 14], dtype=torch.float32)
density = True
res = torch.histogramdd(
    input=input,
    bins=bins,
    range=range,
    weight=weight,
    density=density,
)