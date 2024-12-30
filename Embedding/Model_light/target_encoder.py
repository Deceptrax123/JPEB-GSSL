from torch.nn import Module
from torch_geometric.nn import ChebConv
import torch.nn.functional as F


class TargetEncoder(Module):
    def __init__(self, in_features):
        super(TargetEncoder, self).__init__()

        self.gcn1 = ChebConv(in_channels=in_features,
                             out_channels=250, K=3)
        self.gcn2 = ChebConv(in_channels=250,
                             out_channels=500, K=3)
        self.gcn3 = ChebConv(in_channels=500,
                             out_channels=1000, K=3)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))

        return x
