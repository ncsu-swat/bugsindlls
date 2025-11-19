import torch
data = torch.randn(2, 3)
print(data)
torch.package.PackageExporter(data, './test_package.zip')