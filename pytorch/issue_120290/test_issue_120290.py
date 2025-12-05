import torch
import pickle
import pytest

def test_f():

    tensor = torch.rand([1,2,3],dtype=torch.float32).to(torch.complex32)
    with pytest.raises(KeyError) as e_info:       
        with open('test.pkl', 'wb') as f : pickle.dump(tensor, f)
    print(f'{e_info.type.__name__}: {e_info.value}')