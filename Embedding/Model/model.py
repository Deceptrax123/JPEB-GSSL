from Model.context_encoder import ContextEncoder
from Model.predictor import ContextTargetPredictor
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import dropout_node
from torch.nn import Module
import torch


class EmbeddingModel(Module):
    def __init__(self, num_features, num_targets):
        super(EmbeddingModel, self).__init__()

        self.context_model = ContextEncoder(in_features=num_features)
        self.predictor_model = ContextTargetPredictor(in_features=num_features)
        self.num_targets = num_targets

    def forward(self, G):
        # Consider a context subgraph
        edge_index, _, node_mask = dropout_node(G.edge_index)
        x = self.context_model(G.x, edge_index)

        e_u = []
        for j in range(self.num_targets):
            x = self.predictor_model(x)
            e_u.append(x)

        return torch.tensor(e_u)
