from model import NodeClassifier
from model_light import NodeClassifierLight
from metrics import classification_multiclass_metrics
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.multiprocessing as tmp
from torch import nn
import torch
import os
import random
from dotenv import load_dotenv


def abnormal_feature(ratio):
    g = torch.normal(0, 1, (1000, graph.x.size(1)))
    num_nodes_considered = int(ratio*1000)
    num_deactivated = 1000-num_nodes_considered

    mask = [0]*num_deactivated+[1]*num_nodes_considered
    random.shuffle(mask)
    mask_tensor = torch.tensor(mask)
    random_noise_matrix = mask_tensor.unsqueeze(1)*g

    return torch.mul(graph.x[graph.test_mask], random_noise_matrix)


@torch.no_grad()
def test():
    _, probs = model(graph)

    acc, roc, f1 = classification_multiclass_metrics(
        probs[graph.test_mask], graph.y[graph.test_mask], dataset.num_classes)

    return acc.item(), roc.item(), f1.item()


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    inp_name = input("Enter dataset to be used: ")
    cora_path = os.getenv('Cora')
    pubmed_path = os.getenv('Pubmed')
    citeseer_path = os.getenv('CiteSeer')

    ratio = eval(input('Enter ratio of test nodes to be distorted: '))
    if inp_name == 'cora':
        dataset = Planetoid(root=cora_path, name='Cora')
        weights_path = os.getenv("cora_classification")+"model_1000.pt"
        graph = dataset[0]
        model = NodeClassifier(features=graph.x.size(1),
                               num_classes=dataset.num_classes)
    elif inp_name == 'pubmed':
        dataset = Planetoid(root=pubmed_path, name='PubMed')
        graph = dataset[0]
        model = NodeClassifier(features=graph.x.size(1),
                               num_classes=dataset.num_classes)
        weights_path = os.getenv("pubmed_classification")+"model_550.pt"
    elif inp_name == 'citeseer':
        dataset = Planetoid(root=citeseer_path, name='CiteSeer')
        graph = dataset[0]
        weights_path = os.getenv("citeseer_classification")+"model_165.pt"
        model = NodeClassifierLight(features=graph.x.size(1),
                                    num_classes=dataset.num_classes)

    # Add weights path here
    split = T.RandomNodeSplit(num_val=500, num_test=1000)
    graph = split(graph)
    model.load_state_dict(torch.load(
        weights_path, weights_only=True), strict=True)
    model.eval()

    x_cap = abnormal_feature(ratio)
    graph.x[graph.test_mask] = x_cap

    acc, roc, f1 = test()
    print("Accuracy: ", acc)
    print("AUCROC: ", roc)
    print("F1: ", f1)
