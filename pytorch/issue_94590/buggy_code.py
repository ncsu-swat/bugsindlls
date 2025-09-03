import torch
theta = torch.rand([4, 0, 3508594129291644243, 1, 12, 9], dtype=torch.float32)
size = [21, 6, 34, 108]
align_corners = True
res = torch.affine_grid_generator(
    theta=theta,
    size=size,
    align_corners=align_corners,
)