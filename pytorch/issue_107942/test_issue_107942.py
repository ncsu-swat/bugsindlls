import torch
import pytest
from torch import nn

def test_f():

    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.layer = nn.Dropout(p=1, inplace=False)
        def forward(self, inputs):

            return self.layer(inputs)
        
    input_tensor = torch.randn([1,2]) 
    # Create the model
    mymodel = CustomModel()

    # Forward pass
    output = mymodel(input_tensor) ## No error
    print("Eager result", output)
    mymodel.to('cuda')

    with pytest.raises(RuntimeError) as e_info:
        op_output = torch.compile(mymodel)(input_tensor)
        print("Compiled result", op_output)
    print(f'{e_info.type.__name__}: {e_info.value}')