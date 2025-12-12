import torch
results = dict()
input_1 = torch.rand([5, 0], dtype=torch.float32)
input_2 = torch.rand([5, 0], dtype=torch.float32)

torch.nn.CrossEntropyLoss()(input_1, input_2)
# floating point exception