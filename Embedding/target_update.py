import torch
from torch_geometric import nn


def ema_target_weights(target_encoder, context_encoder, sf=0.9):
    for (m1, m2) in zip(target_encoder.modules(), context_encoder.modules()):
        if isinstance(m1, (nn.ChebConv)):
            m1.weight.data = (m2.weight.data*sf)+(m1.weight.data*(1-sf))
