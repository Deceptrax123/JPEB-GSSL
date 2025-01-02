from model import NodeClassifier
from model_light import NodeClassifierLight
from metrics import classification_multiclass_metrics
from torch_geometric.utils import dropout_node
from torch_geometric.datasets import Amazon, Coauthor
import torch_geometric.transforms as T
import torch.multiprocessing as tmp
from torch import nn
import torch
import os
import random
from dotenv import load_dotenv


def abnormal_feature(edge_index):

    changed_edges, _, node_mask = dropout_node(edge_index, p=0.1)

    return changed_edges, node_mask


@torch.no_grad()
def test():
    _, probs = model(graph)

    acc, roc, f1 = classification_multiclass_metrics(
        probs, graph.y, dataset.num_classes)

    return acc.item(), roc.item(), f1.item()


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    inp_name = input("Enter dataset to be used: ")
    cora_path = os.getenv('Cora')
    pubmed_path = os.getenv('Pubmed')
    citeseer_path = os.getenv('CiteSeer')
    computers_path = os.getenv('Computers')
    photos_path = os.getenv('Photo')
    cs_path = os.getenv('CS')

    if inp_name == 'computers':
        dataset = Amazon(root=computers_path, name='Computers')
        graph = dataset[0]
        weights_path = os.getenv("computer_classification")+"model_70.pt"
        model = NodeClassifier(features=graph.x.size(1),
                               num_classes=dataset.num_classes)
    elif inp_name == 'photos':
        dataset = Amazon(root=photos_path, name='Photo')
        graph = dataset[0]
        weights_path = os.getenv("photo_classification")+"model_65.pt"
        model = NodeClassifier(features=graph.x.size(1),
                               num_classes=dataset.num_classes)
    elif inp_name == 'cs':
        dataset = Coauthor(root=cs_path, name='CS')
        graph = dataset[0]
        weights_path = os.getenv('CS_classification')+"model_350.pt"
        model = NodeClassifierLight(features=graph.x.size(1),
                                    num_classes=dataset.num_classes)

    # Add weights path here
    model.load_state_dict(torch.load(
        weights_path, weights_only=True), strict=True)
    model.eval()

    edges, node_mask = abnormal_feature(graph.edge_index)
    graph.x = node_mask.unsqueeze(1)*graph.x
    graph.edge_index = edges

    acc, roc, f1 = test()
    print("Accuracy: ", acc)
    print("AUCROC: ", roc)
    print("F1: ", f1)
