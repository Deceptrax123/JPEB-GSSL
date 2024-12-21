from torch.nn import Tanh, Module
from torch_geometric.nn import GCNConv, GraphNorm
import torch.nn.functional as F


class ContextTargetPredictor(Module):
    def __init__(self, dims):
        super(ContextTargetPredictor, self).__init__()

        self.gc1 = GCNConv(
            in_channels=dims, out_channels=dims//2, normalize=False)
        self.norm = GraphNorm(dims//2)
        self.tanh1 = Tanh()

        self.gc2 = GCNConv(out_channels=dims//2,
                           in_channels=dims//2, normalize=False)
        self.tanh2 = Tanh()

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index=edge_index)
        x = self.norm(x)
        x = self.tanh1(x)

        x = self.gc2(x, edge_index=edge_index)
        x = self.tanh2(x)

        return x
