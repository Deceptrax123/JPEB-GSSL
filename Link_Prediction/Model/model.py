from torch_geometric.nn import ChebConv
from torch.nn import Module
import torch.nn.functional as F


class ContextEncoder(Module):
    def __init__(self, in_features):
        super(ContextEncoder, self).__init__()

        self.gcn1 = ChebConv(in_channels=in_features,
                             out_channels=in_features*2, K=3)
        self.gcn2 = ChebConv(in_channels=in_features*2,
                             out_channels=in_features*4, K=3)
        self.gcn3 = ChebConv(in_channels=in_features*4,
                             out_channels=in_features*4, K=3)

    def forward(self, x, edge_index):

        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))

        return x


class LinkPredictor(Module):
    def __init__(self, features):
        self.encoder = ContextEncoder(in_features=features)

    def encode(self, graph):
        x, edge_index = graph.x, graph.edge_index

        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        t = (z[edge_label_index[0]]*z[edge_label_index[1]].sum(dim=-1))

        return t, F.sigmoid(t)

    def decode_all(self, z):
        prob_adj = z@z.t()

        return (prob_adj > 0).nonzero(as_tuple=False).t()
