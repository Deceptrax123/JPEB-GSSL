from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from torch_geometric.utils import dropout_node
from Model.model import EmbeddingModel
from Model.target_encoder import TargetEncoder
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


def train_epoch():
    # View augmentations take place in the Embedding Class.
    encoder_embeddings = embedding_model(graph)

    # Generate Targets based on Bernoulli Distribution
    target_loss = 0
    embedding_model.zero_grad()
    for _ in range(num_targets):
        target_embedding = target_encoder(graph)
        _, _, node_mask = dropout_node(graph.edge_index)

        # Mask based features
        target_features = node_mask.unsqueeze(1)*target_embedding
        encoder_mask_features = node_mask.unsqueeze(
            1)*encoder_embeddings  # Pass positional information to Nodes

        target_loss += l2_loss(target_features, encoder_mask_features)

    loss = target_loss/num_targets
    loss.backward()

    optimizer.step()

    return loss


def training_loop():
    for epoch in range(EPOCHS):
        embedding_model.train()
        train_loss = train_epoch()

        embedding_model.eval()

        wandb.log({
            "Embedding Loss": train_loss
        })

        # Save weights
        if (epoch+1) % 2 == 0:
            save_encoder_weights = f"Embedding/Weights/Encoder/Run_1/model_{
                epoch+1}.pt"
            save_arch_weights = f"Embedding/Weights/Architecture/Run_1/model_{
                epoch+1}.pt"

            torch.save(embedding_model.state_dict(), save_arch_weights)
            torch.save(embedding_model.context_model.state_dict(),
                       save_encoder_weights)
        scheduler.step()


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

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }
    num_targets = 3
    embedding_model = EmbeddingModel(
        num_features=graph.x.size(1), num_targets=3)
    target_encoder = TargetEncoder(in_features=graph.x.size(1))

    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    graph = split(graph)

    l2_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=embedding_model.parameters(), lr=LR, betas=BETAS, eps=EPSILON)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, verbose=True)

    wandb.init(
        project="Joint Graph embedding development",
        config={
            "Method": "Generative",
            "Dataset": "Planetoid and Amazon"
        }
    )
