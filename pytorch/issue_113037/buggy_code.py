import torch
import numpy as np

args = {
    'input': torch.from_numpy(np.array(np.random.uniform(-1000_000, 1000_000, [8, 0])).astype('float16')),
    'other': torch.from_numpy(np.array(np.random.uniform(-1000_000, 1000_000, [0, 8, 0])).astype('float16')),
    'out': torch.from_numpy(np.array(np.random.uniform(-1000_000, 1000_000, [0])).astype('float32')),
    'rounding_mode': 'floor'
}
torch.div(**args)