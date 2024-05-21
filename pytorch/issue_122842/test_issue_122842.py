import torch
import torch.nn as nn
import pytest

# author a model.
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 8)
        self.relu4 = nn.ReLU()
        
    def forward(self, x):
        layer1 = self.relu1(self.fc1(x))
        layer2 = self.relu2(self.fc2(layer1))
        layer3 = self.relu3(self.fc3(layer2))
        y      = self.relu4(self.fc4(layer3))
        return y
    
    def name(self):
        return "MLP"

def f():
    model = MLPNet().eval()

    # generate a uniform distribution of data.
    n_batches = 100
    x_in = torch.distributions.uniform.Uniform(-1, 1).sample([n_batches, 64, 64])

    # export the module (this will annotate the graph with node names such as
    # "fc1_weight", "fc1_bias", ...), but not argument names (ex: "x").
    m_export = torch.export.export(model, (x_in[0,:],))
    print('---------------')
    print('torch.export():')
    print('---------------')
    m_export.module().graph.print_tabular()
    print()
    return list(m_export.module().graph.nodes)

def test_f():
    issue_no = '122842'
    print('Pytorch issue no.', issue_no)
    
    node_names = [node.name for node in f()]
    assert "1_x" not in node_names # "1_x" is not captured
