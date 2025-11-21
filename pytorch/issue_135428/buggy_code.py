import torch
import numpy as np

p = 196

a = torch.tensor([[0.54269608+0.j,  1.57257892+0.j],
                  [0.14092422+0.j,  0.69102135+0.j],
                  [0.80219698+0.j,  0.12309907+0.j],
                  [0.07455064+0.j, -0.43134838+0.j],
                  [0.98688694+0.j,  2.097082 + 0.j]])

cpu = np.allclose(torch.nn.functional.normalize(a, p=p).real, torch.nn.functional.normalize(a.real, p=p))
# false

gpu =  np.allclose(torch.nn.functional.normalize(a.cuda(), p=p).real.cpu(), torch.nn.functional.normalize(a.real.cuda(), p=p).cpu())
# true

print("cpu:", cpu)
print("gpu:", gpu)

print(torch.nn.functional.normalize(a, p=p))
# tensor([[        nan+nanj,         nan+nanj],
#         [ 2.0394e-01+0.j,  1.0000e+00+0.j],
#         [ 1.0000e+00+0.j,  1.5345e-01+0.j],
#         [ 7.4551e+10+0.j, -4.3135e+11+0.j],
#         [ 0.0000e+00+0.j,  0.0000e+00+0.j]])