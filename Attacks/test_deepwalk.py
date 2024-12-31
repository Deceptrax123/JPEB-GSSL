from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import DeepWalk, Node2Vec
from deeprobust.graph.global_attack import NodeEmbeddingAttack
import numpy as np

dataset_str = 'cora_ml'
data = Dataset(root='/tmp/', name=dataset_str, seed=15)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

attacker = NodeEmbeddingAttack()
attacker.attack(adj, attack_type="remove", n_perturbations=1000)
modified_adj = attacker.modified_adj

# train defense model
print("Test Node2vec on clean graph")
model = Node2Vec()
model.fit(adj)
model.evaluate_node_classification(labels, idx_train, idx_test)
# model.evaluate_node_classification(labels, idx_train, idx_test, lr_params={"max_iter": 1000})

print("Test Node2vec on attacked graph")
model.fit(modified_adj)
model.evaluate_node_classification(labels, idx_train, idx_test)
