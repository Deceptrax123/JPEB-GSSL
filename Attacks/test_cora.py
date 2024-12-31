from deeprobust.graph.data import Dpr2Pyg, Dataset
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from metrics import classification_multiclass_metrics
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from model import NodeClassifier
import torch
from dotenv import load_dotenv
import os


@torch.no_grad()
def test():
    _, probs = model(graph)

    acc, roc, f1 = classification_multiclass_metrics(
        probs[graph.test_mask], graph.y[graph.test_mask], data.num_classes)

    return acc.item(), roc.item(), f1.item()


load_dotenv('.env')
path = os.getenv('Cora')
cora_weights = os.getenv('cora_classification')+"model_1000.pt"

data = Planetoid(root=path, name='Cora')
data_dr = Dataset(root="../", name='cora', seed=15)
adj, features, labels = data_dr.adj, data_dr.features, data_dr.labels

attacker = NodeEmbeddingAttack()
attacker.attack(adj, attack_type='remove', n_perturbations=250)
modified_adj = attacker.modified_adj
edge_index_modifed = dense_to_sparse(torch.tensor(modified_adj.toarray()))
graph = data[0]
graph.edge_index = edge_index_modifed[0]

# Test on Cora
model = NodeClassifier(features=graph.x.size(1), num_classes=data.num_classes)
model.load_state_dict(torch.load(cora_weights, weights_only=True), strict=True)
model.eval()

acc, roc, f1 = test()
print("Accuracy: ", acc)
print("AUCROC: ", roc)
print("F1: ", f1)
