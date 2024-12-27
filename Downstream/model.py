from torch.nn import Module
from torch_geometric.nn import ChebConv
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


class NodeClassifier(Module):
    def __init__(self, features, num_classes):
        super(NodeClassifier, self).__init__()

        self.encoder = ContextEncoder(in_features=features)
        self.classifier = ChebConv(
            in_channels=4*features, out_channels=num_classes, K=3)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.encoder(x, edge_index)
        x = self.classifier(x)

        return x
