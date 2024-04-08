import torch
torch.utils.collect_env.main()
torch._C._nn.replication_pad2d(torch.rand([2]), padding=[])