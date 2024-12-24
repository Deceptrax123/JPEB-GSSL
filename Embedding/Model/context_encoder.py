from torch.nn import Module
from torch_geometric.nn import ChebConv
import torch.nn.functional as F


class ContextEncoder(Module):
    def __init__(self, in_features):
        super(ContextEncoder, self).__init__()

        self.gcn1 = ChebConv(in_channels=in_features,
                             out_channels=in_features*2)
        self.gcn2 = ChebConv(in_channels=in_features*2,
                             out_channels=in_features*4)
        self.gcn3 = ChebConv(in_channels=in_features*4,
                             out_channels=in_features*4)

    def forward(self, x, edge_index):

        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))

        return x
