from model import ContextEncoderLite
from torch_geometric.datasets import Planetoid, Amazon
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, v_measure_score
import torch_geometric.transforms as T
import torch.multiprocessing as tmp
from matplotlib.colors import ListedColormap
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from dotenv import load_dotenv


@torch.no_grad()
def cluster():
    z = model(graph.x, graph.edge_index).numpy()
    # projected_2d = tsne_transform.fit_transform(z)
    projected_2d = tsne_transform.fit_transform(z[graph.test_mask])

    x_min, x_max = projected_2d[:, 0].min()-1, projected_2d[:, 0].max()+1
    y_min, y_max = projected_2d[:, 1].min()-1, projected_2d[:, 1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    kmeans_transform.fit(projected_2d.astype('double'))

    z_kmeans = kmeans_transform.predict(
        np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z_kmeans, cmap=cmap_light, alpha=0.6)

    plt.scatter(projected_2d[:, 0], projected_2d[:, 1],
                c=kmeans_transform.labels_, s=50, edgecolor='k', cmap='viridis')

    plt.show()


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
        cmap_light = ListedColormap(
            ["#ADD8E6", "#90EE90", "#F08080", "#FFB6C1", "#FFFFE0", "#E0FFFF", "#D8BFD8"])
        dataset = Planetoid(root=cora_path, name='Cora')
        graph = dataset[0]
        weights_path = os.getenv("cora_encoder")+"model_25900.pt"
    elif inp_name == 'pubmed':
        cmap_light = ListedColormap(['#ADD8E6', '#FFB6C1', '#90EE90'])
        dataset = Planetoid(root=pubmed_path, name='PubMed')
        graph = dataset[0]
        weights_path = os.getenv("pubmed_encoder")+"model_400.pt"
    elif inp_name == 'computers':
        cmap_light = ListedColormap(['#ADD8E6', '#FFB6C1', '#90EE90', '#FFFFE0',
                                    '#E6E6FA', '#F08080', '#FFDAB9', '#D8BFD8', '#E0FFFF', '#FAFAD2'])
        dataset = Amazon(root=computers_path, name='Computers')
        graph = dataset[0]
        weights_path = os.getenv("computer_encoder")+"model_30.pt"
    elif inp_name == 'photos':
        cmap_light = ListedColormap(['#ADD8E6', '#FFB6C1', '#90EE90', '#FFFFE0', '#E6E6FA',
                                     '#F08080', '#FFDAB9', '#D8BFD8'])
        dataset = Amazon(root=photos_path, name='Photo')
        graph = dataset[0]
        weights_path = os.getenv("photo_encoder")+"model_265.pt"

    model = ContextEncoderLite(in_features=graph.x.size(1))
    model.load_state_dict(torch.load(
        weights_path, weights_only=True), strict=True)
    model.eval()
    split = T.RandomNodeSplit(num_test=1000, num_val=500)
    graph = split(graph)

    tsne_transform = TSNE(
        n_components=2, learning_rate='auto', init='random', perplexity=5)
    kmeans_transform = KMeans(n_clusters=dataset.num_classes, random_state=0)

    pca_transform = PCA(n_components=2)

    cluster()
