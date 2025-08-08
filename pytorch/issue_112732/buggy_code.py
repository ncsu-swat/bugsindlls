import torch
import numpy as np

args = {
    'input1': torch.from_numpy(np.array(np.random.uniform(-1000_000, 1000_000, [0, 0])).astype('float16')), 
    'input2': torch.from_numpy(np.array(np.random.randint(-1000_000, 1000_000, [0])).astype('int8')),
    'margin': 6.4513009, 
    'reduce': True, 
    'reduction': 'mean', 
    'size_average': True, 
    'target': torch.from_numpy(np.array(np.random.uniform(-1000_000, 1000_000, [0])).astype('float32')), 
}

torch.nn.functional.cosine_embedding_loss(**args)