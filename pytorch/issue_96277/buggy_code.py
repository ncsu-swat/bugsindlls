import torch
input = torch.rand([0, 9, 1402528952189899978], dtype=torch.float32)
output_size = [1]
res = torch.adaptive_max_pool1d(
    input=input,
    output_size=output_size,
)