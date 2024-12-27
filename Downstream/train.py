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


def train_epoch():
    model.zero_grad()

    logits, probs = model(graph)
    loss = objective_function(
        logits[graph.train_mask], graph.y[graph.train_mask])

    loss.backward()
    optimizer.step()

    # Metric
    acc, roc, f1 = classification_multiclass_metrics(
        probs[graph.train_mask], graph.y[graph.train_mask], dataset.num_classes)

    return loss.item(), acc.item(), roc.item(), f1.item()


def val_epoch():
    logits, probs = model(graph)
    loss = objective_function(
        logits[graph.val_mask], graph.y[graph.val_mask])

    acc, roc, f1 = classification_multiclass_metrics(
        probs[graph.val_mask], graph.y[graph.val_mask], dataset.num_classes)

    return loss.item(), acc.item(), roc.item(), f1.item()


def training_loop():
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc, train_roc, train_f1 = train_epoch()

        model.eval()
        with torch.no_grad():
            test_loss, test_acc, test_roc, test_f1 = val_epoch()
            print("Epoch: ", epoch+1)
            print("Train Loss: ", train_loss)
            print("Train Accuracy: ", train_acc)
            print("Train ROC: ", train_roc)
            print("Train F1: ", train_f1)
            print("Test Loss: ", test_loss)
            print("Test Accuracy: ", test_acc)
            print("Test ROC: ", test_roc)
            print("Test F1: ", test_f1)

            wandb.log({
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Train ROC": train_roc,
                "Train F1": train_f1,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "Test ROC": test_roc,
                "Test F1": test_f1
            })

            if (epoch+1) % 5 == 0:
                save_path = os.getenv(
                    "cora_classification")+f"model_{epoch+1}.pt"

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

    # params = {
    #     'batch_size': 32,
    #     'shuffle': True,
    #     'num_workers': 0
    # }
    model = NodeClassifier(features=graph.x.size(1),
                           num_classes=dataset.num_classes)

    objective_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=LR, betas=BETAS, eps=EPSILON)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10)

    weights_path = os.getenv("cora_encoder")+"model_140.pt"
    model.encoder.load_state_dict(torch.load(
        weights_path, weights_only=True), strict=True)
    init_weights(model.classifier)

    wandb.init(
        project="Joint Graph embedding downstream tests",
        config={
            "Method": "Generative",
            "Dataset": "Planetoid and Amazon"
        }
    )

    training_loop()
