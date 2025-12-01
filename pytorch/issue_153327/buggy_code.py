import torch

print(torch.__version__)

tensor = torch.tensor([], dtype=torch.float16).reshape(0,0,9)
upscale_factor = 4323455642275676160

torch.pixel_shuffle(
    tensor,
    upscale_factor
)