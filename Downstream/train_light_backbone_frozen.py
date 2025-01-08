from torch_geometric.graphgym import init_weights
from model_light import NodeClassifier
from metrics import classification_multiclass_metrics, classification_binary_metrics
from torch_geometric.datasets import Planetoid, Coauthor
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
    for epoch in range(20000, EPOCHS):
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

            if (epoch+1) % 1000 == 0:
                save_path = os.getenv(
                    "cora_frozen")+f"model_{epoch+1}.pt"

                torch.save(model.state_dict(), save_path)

            scheduler.step()


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    inp_name = input("Enter dataset to be used: ")
    citeseer_path = os.getenv('CiteSeer')
    cs_path = os.getenv('CS')

    if inp_name == 'citeseer':
        dataset = Planetoid(root=citeseer_path, name='Citeseer')
        graph = dataset[0]
        weights_path = os.getenv("citeseer_encoder")+"model_500.pt"

        split_function = T.RandomNodeSplit(num_val=500, num_test=1000)
        graph = split_function(graph)
    elif inp_name == 'cs':
        dataset = Coauthor(root=cs_path, name='CS')
        graph = dataset[0]
        weights_path = os.getenv("CS_encoder")+"model_85.pt"

        split_function = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
        graph = split_function(graph)

    model = NodeClassifier(features=graph.x.size(1),
                           num_classes=dataset.num_classes)
    # model.encoder.load_state_dict(torch.load(
    #     weights_path, weights_only=True), strict=True)
    # init_weights(model.classifier)

    restart_ckpt = os.getenv('cora_frozen')+"model_20000.pt"
    model.load_state_dict(torch.load(
        restart_ckpt, weights_only=True), strict=True)

    for param in model.encoder.parameters():
        # freeze the backbone
        param.requires_grad = False

    objective_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=LR, betas=BETAS, eps=EPSILON)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100)

    wandb.init(
        project="Joint Graph embedding downstream tests",
        config={
            "Method": "Generative",
            "Dataset": "Planetoid and Amazon"
        }
    )

    training_loop()
