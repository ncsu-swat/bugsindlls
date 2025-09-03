import torch
import numpy as np
print(torch.__version__)
input = torch.rand([12, 10, 0, 8, 7], dtype=torch.float32)
res = torch.fbgemm_pack_gemm_matrix_fp16(input)
print(res)