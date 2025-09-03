import torch
input = torch.rand([0, 14, 6068228052691422846], dtype=torch.float32)
lengths = torch.rand([11, 14, 14, 2], dtype=torch.float32)
batch_first = True
torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first)