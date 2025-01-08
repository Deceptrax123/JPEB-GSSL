from torch_geometric.utils import dropout_node
from Model_light.model import EmbeddingModel
from Model_light.target_encoder import TargetEncoder
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from hyperparameters import LR, EPSILON, EPOCHS, BETAS
from target_update import ema_target_weights
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
    for i in range(num_targets):
        target_embedding = target_encoder(graph)
        _, _, node_mask = dropout_node(graph.edge_index, p=0.01)

        # Mask based features
        target_features = node_mask.unsqueeze(1)*target_embedding
        encoder_mask_features = node_mask.unsqueeze(
            1)*encoder_embeddings[i]  # Pass positional information to Nodes

        target_loss += l2_loss(encoder_mask_features, target_features)

        del target_embedding, target_features, node_mask, encoder_mask_features
    del encoder_embeddings

    loss = target_loss/num_targets
    loss.backward()

    optimizer.step()
    # Update target encoder weight
    ema_target_weights(target_encoder, embedding_model.context_model)

    return loss


def training_loop():
    for epoch in range(EPOCHS):
        embedding_model.train()
        # target_encoder.requires_grad_ = False
        train_loss = train_epoch()

        embedding_model.eval()

        wandb.log({
            "Embedding Loss": train_loss.item()
        })

        print("Epoch: ", epoch+1)
        print("Embedding Loss: ", train_loss.item())

        # Save weights
        if (epoch+1) % 50 == 0:
            save_encoder_weights = os.getenv(
                "pubmed_encoder")+f"model_{epoch+1}.pt"

            torch.save(embedding_model.context_model.state_dict(),
                       save_encoder_weights)
        # scheduler.step()


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
    physics_path = os.getenv('Physics')
    cs_path = os.getenv('CS')

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

    num_targets = 3
    embedding_model = EmbeddingModel(
        num_features=graph.x.size(1), num_targets=3)
    target_encoder = TargetEncoder(in_features=graph.x.size(1))

    l2_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=embedding_model.parameters(), lr=LR, betas=BETAS, eps=EPSILON)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10)

    wandb.init(
        project="Joint Graph embedding development",
        config={
            "Method": "Generative",
            "Dataset": "Planetoid and Amazon"
        }
    )

    training_loop()
