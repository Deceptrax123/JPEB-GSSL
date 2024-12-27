from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from torch_geometric.utils import dropout_node
from model import NodeClassifier
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


def training_loop():
    pass


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
        graph = Planetoid(root=cora_path, name='Cora')
    elif inp_name == 'pubmed':
        graph = Planetoid(root=pubmed_path, name='PubMed')
    elif inp_name == 'citeseer':
        graph = Planetoid(root=citeseer_path, name='CiteSeer')
    elif inp_name == 'computers':
        graph = Amazon(root=computers_path, name='Computers')
    elif inp_name == 'photos':
        graph = Amazon(root=photos_path, name='Photo')

    # params = {
    #     'batch_size': 32,
    #     'shuffle': True,
    #     'num_workers': 0
    # }
    model = NodeClassifier(features=graph.x.size(1),
                           num_classes=graph.num_classes)

    l2_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=LR, betas=BETAS, eps=EPSILON)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10)

    init_weights(model)

    wandb.init(
        project="Joint Graph embedding downstream tests",
        config={
            "Method": "Generative",
            "Dataset": "Planetoid and Amazon"
        }
    )
