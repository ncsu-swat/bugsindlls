import requests
import pickle
import torch
import torch.nn as nn
import traceback
import os
import numpy as np

def test_f():

    def load_pickle_from_url(url) :
        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            # Load the content of the file
            data = pickle.loads(response.content)

            return data
        else:
            print("Failed to download the file")
            return None

    pickle_url = "https://github.com/GwiHwan-Go/repo/raw/main/issues/pickles/issue_117544.pkl"
    inputs = load_pickle_from_url(pickle_url)

    if inputs is None:
        raise ValueError("Pickle file could not be downloaded. Please check the URL.")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, out):   
            out = torch.sinh(input=out)        
            out = torch.relu(input=out)        
            return out

    x = inputs[0]['input']
    print(x)
    model = Model().to(torch.device('cpu'))
    eag = model(x)
    print(eag)
    opt = torch.compile(model.forward, mode='max-autotune')(x)
    print(opt)
    assert not torch.allclose(eag.to('cpu'), opt.to('cpu'), rtol=1e-3, atol=1e-3, equal_nan=True)