from Model_light.context_encoder import ContextEncoder
from Model_light.predictor import ContextTargetPredictor
from torch_geometric.utils import dropout_node
from torch.nn import Module
import torch


class EmbeddingModel(Module):
    def __init__(self, num_features, num_targets):
        super(EmbeddingModel, self).__init__()

        self.context_model = ContextEncoder(in_features=num_features)
        self.predictor_model = ContextTargetPredictor(dims=1000)
        self.num_targets = num_targets

    def forward(self, G):
        # Consider a context subgraph
        edge_index, _, _ = dropout_node(
            G.edge_index, p=0.05)  # Bernoulli Distribution
        x = self.context_model(G.x, edge_index)

        e_u = []
        for _ in range(self.num_targets):
            v = self.predictor_model(x, edge_index)
            e_u.append(v)

        return e_u
