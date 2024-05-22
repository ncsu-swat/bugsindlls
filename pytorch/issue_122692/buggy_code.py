import torch

class MyModule(torch.nn.Module):
    def forward(self, input_id):
        if input_id.is_complex():
            return torch.view_as_real(input_id)
        else:
            return input_id

my_mod = MyModule()
my_mod = torch.compile(my_mod)

out = my_mod(torch.ones(5))
