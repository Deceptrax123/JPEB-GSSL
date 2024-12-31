from model_light import NodeClassifier
from metrics import classification_multiclass_metrics, classification_binary_metrics
from torch_geometric.datasets import Planetoid, Coauthor
import torch_geometric.transforms as T
import torch.multiprocessing as tmp
from torch import nn
import torch
import os
from dotenv import load_dotenv


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
    citeseer_path = os.getenv('CiteSeer')
    cs_path = os.getenv('CS')

    if inp_name == 'citeseer':
        dataset = Planetoid(root=citeseer_path, name='CiteSeer')
        graph = dataset[0]
        weights_path = os.getenv("citeseer_classification")+"model_85.pt"
    elif inp_name == 'cs':
        dataset = Coauthor(root=cs_path, name='CS')
        graph = dataset[0]
        weights_path = os.getenv('CS_classification')+"model_350.pt"

    # split_function = T.RandomNodeSplit(num_val=500, num_test=1000)
    # graph = split_function(graph)

    model = NodeClassifier(features=graph.x.size(1),
                           num_classes=dataset.num_classes)

    # Add weights path here
    model.load_state_dict(torch.load(
        weights_path, weights_only=True), strict=True)
    model.eval()

    acc, roc, f1 = test()
    print("Accuracy: ", acc)
    print("AUCROC: ", roc)
    print("F1: ", f1)
