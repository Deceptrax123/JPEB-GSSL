from torch_geometric.graphgym import init_weights
from Model.model import LinkPredictor
from torch_geometric.datasets import Planetoid, Amazon
from hyperparameters import LR, EPSILON, EPOCHS, BETAS
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
import torch.multiprocessing as tmp
from torch import nn
import torch
import os
import wandb
import gc
from dotenv import load_dotenv


def train_epoch():
    model.zero_grad()

    z = model.encode(train_data)

    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1),
        method='sparse'
    )

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index], dim=-1
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    output, probs = model.decode(z, edge_label_index).view(-1)
    loss = objective_function(output, edge_label)
    loss.backward()
    optimizer.step()

    auc_score = roc_auc_score(edge_label.numpy(), probs.numpy())

    return loss.item(), auc_score


def val_epoch():
    z = model.encode(val_data)

    output, probs = model.decode(z, val_data.edge_label_index).view(-1)
    loss = objective_function(output, val_data.edge_label)

    auc_score = roc_auc_score(val_data.edge_label.numpy(), probs.cpu().numpy())

    return loss.item(), auc_score


def training_loop():
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_auc = train_epoch()

        model.eval()
        with torch.no_grad():
            val_loss, val_auc = val_epoch()

            print("Epoch: ", epoch+1)
            print("Train Loss: ", train_loss)
            print("Train AUC: ", train_auc)
            print("Validation Loss: ", val_loss)
            print("Validation AUC: ", val_auc)

            wandb.log({
                "Train Loss": train_loss,
                "Train AUC": train_auc,
                "Validation Loss": val_loss,
                "Validation AUC: ": val_auc
            })

            if (epoch+1) % 5 == 0:
                save_path = os.getenv('cora_link')+f"model_{epoch+1}.pt"

                torch.save(model.state_dict(), save_path)


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
        weights_path = os.getenv("cora_encoder")+"model_140.pt"
    elif inp_name == 'pubmed':
        dataset = Planetoid(root=pubmed_path, name='PubMed')
        graph = dataset[0]
        weights_path = os.getenv("pubmed_encoder")+"model_130.pt"
    elif inp_name == 'citeseer':
        weights_path = os.getenv("citeseer_encoder")+"model_140.pt"
    elif inp_name == 'computers':
        dataset = Amazon(root=computers_path, name='Computers')
        graph = dataset[0]
        weights_path = os.getenv("computer_encoder")+"model_30.pt"
    elif inp_name == 'photos':
        dataset = Amazon(root=photos_path, name='Photo')
        graph = dataset[0]
        weights_path = os.getenv("photo_encoder")+"model_265.pt"

    edge_split = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0
    )

    train_data, val_data, _ = edge_split(graph)

    model = LinkPredictor(features=graph.x.size(1))
    encoder_weights = torch.load(weights_path, weights_only=True)
    model.encoder.load_state_dict(encoder_weights, strict=True)

    objective_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), betas=BETAS, eps=EPSILON, lr=LR)

    wandb.init(
        project="Joint Graph Embedding Link Prediction Tests",
        config={
            "Method": "Generative",
            "Dataset": "Planetoid and Amazon"
        }
    )

    training_loop()
