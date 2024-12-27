from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from model import NodeClassifier
from metrics import classification_multiclass_metrics, classification_binary_metrics
from torch_geometric.datasets import Planetoid, Amazon
from hyperparameters import LR, EPSILON, EPOCHS, BETAS
import torch_geometric.transforms as T
import torch.multiprocessing as tmp
from torch import nn
import torch
import os
import wandb
import gc
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
    cora_path = os.getenv('Cora')
    pubmed_path = os.getenv('Pubmed')
    citeseer_path = os.getenv('CiteSeer')
    computers_path = os.getenv('Computers')
    photos_path = os.getenv('Photo')

    if inp_name == 'cora':
        dataset = Planetoid(root=cora_path, name='Cora')
        graph = dataset[0]
    elif inp_name == 'pubmed':
        graph = Planetoid(root=pubmed_path, name='PubMed')
    elif inp_name == 'citeseer':
        graph = Planetoid(root=citeseer_path, name='CiteSeer')
    elif inp_name == 'computers':
        graph = Amazon(root=computers_path, name='Computers')
    elif inp_name == 'photos':
        graph = Amazon(root=photos_path, name='Photo')

    split_function = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    graph = split_function(graph)

    model = NodeClassifier(features=graph.x.size(1),
                           num_classes=dataset.num_classes)
    model.load_state_dict()  # Add weights path here
    model.eval()

    acc, roc, f1 = test()
    print("Accuracy: ", acc)
    print("AUCROC: ", roc)
    print("F1: ", f1)
