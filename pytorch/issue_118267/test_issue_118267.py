import torch
import pytest
import torch.nn as nn
import os, pickle
import numpy as np 
import requests

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

    pickle_url = "https://github.com/GwiHwan-Go/repo/raw/main/issues/pickles/torch_issue_118267.pkl"
    inputs = load_pickle_from_url(pickle_url)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, out):
            out = torch.acosh(input=input, out=out)        
            return out

    out1 = torch.rand([10, 9], dtype=torch.float32)
    out2 = out1.clone()

    model = Model().to(torch.device('cpu'))
    eag = model(inputs[0]['input'], out1)
    opt = torch.compile(model.forward)(inputs[0]['input'], out2)

    assert not torch.allclose(eag, opt, rtol=1e-3, atol=1e-3, equal_nan=True)