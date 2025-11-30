import torch

print(torch.__version__)

input_tensor = torch.tensor([[         float('nan'),          float('nan'),          float('nan'),          float('nan'),          float('nan'),
                  float('nan'),          float('nan'), -7.4786e+240, -7.4786e+240, -8.7289e+240,
         -7.4786e+240, -7.4786e+240, -7.4786e+240, -7.4786e+240],
        [-7.4786e+240, -7.4786e+240,          float('nan'),          float('nan'), -7.4786e+240,
         -7.4786e+240, -7.4786e+240, -7.4786e+240, -7.4786e+240,          float('nan'),
                  float('nan'),          float('nan'), -7.8480e+298, -7.8459e+298],
        [-7.4786e+240,          float('nan'),          float('nan'),          float('nan'),          float('nan'),
                  float('nan'), -5.4902e+303, -1.2503e+241, -7.4786e+240,  2.6462e-260,
          2.6462e-260,  2.6462e-260,  2.6462e-260, -7.4786e+240]],
       dtype=torch.float64)
running_mean = torch.tensor([-7.4786e+240, -7.4786e+240, -7.4786e+240, -7.4835e+240, -7.4786e+240,
        -7.4786e+240, -7.4786e+240, -7.4786e+240, -7.4786e+240, -7.4786e+240,
        -7.4786e+240], dtype=torch.float64)
running_var = torch.tensor([-7.4786e+240, -7.4781e+240, -7.4786e+240, -7.4786e+240,   1.9410e-80,
        -7.4774e+240, -7.4786e+240, -7.4786e+240, -7.8459e+298,  1.0869e-322,
          0.0000e+00], dtype=torch.float64)
momentum = 0.1

torch.batch_norm_update_stats(
    input_tensor,
    running_mean,
    running_var,
    momentum,
)