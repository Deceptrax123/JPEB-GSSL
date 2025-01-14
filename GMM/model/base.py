from typing import Iterator, List, Optional
from abc import ABC, abstractmethod

import torch
from torch.distributions.utils import logits_to_probs

from GMM.visualize import plot_data_and_model
from GMM.utils import make_random_scale_trils


class MixtureModel(ABC, torch.nn.Module):
    """
    Base model for mixture models

    :param num_components: Number of component distributions
    :param num_dims: Number of dimensions being modeled
    :param init_radius: L1 radius within which each component mean should
        be initialized, defaults to 1.0
    :param init_mus: mean values to initialize model with, defaults to None
    """

    def __init__(
        self,
        num_components: int,
        num_dims: int,
        mixture_lr: float,
        component_lr: float,
        init_radius: float = 1.0,
        init_mus: List[List[float]] = None,
    ):
        super().__init__()
        self.num_components = num_components
        self.num_dims = num_dims

        self.logits = torch.nn.Parameter(torch.zeros(num_components, ))

        self.mus = torch.nn.Parameter(
            torch.tensor(init_mus, dtype=torch.float32)
            if init_mus is not None
            else torch.rand(num_components, num_dims).uniform_(-init_radius, init_radius)
        )

        # lower triangle representation of (symmetric) covariance matrix
        self.scale_tril = torch.nn.Parameter(
            make_random_scale_trils(num_components, num_dims))

        self.mixture_optimizer = torch.optim.Adam(
            self.mixture_parameters(), lr=mixture_lr)
        self.mixture_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.mixture_optimizer, T_0=25
        )
        self.components_optimizer = torch.optim.Adam(
            self.component_parameters(), lr=component_lr)
        self.components_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.components_optimizer, T_0=25
        )

    def mixture_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.logits])

    def get_probs(self) -> torch.Tensor:
        return logits_to_probs(self.logits)

    def fit(self):
        # optimize
        # # forward
        # loss = self(data)
        # print(loss)

        # # log and visualize
        # if log_freq is not None:
        #     print(f" Loss: {loss.item():.2f}")
        # if visualize:
        #     plot_data_and_model(data, self)

        # Step
        self.mixture_optimizer.step()
        self.mixture_scheduler.step()
        self.components_optimizer.step()
        self.components_scheduler.step()

        # constrain parameters
        self.constrain_parameters()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def constrain_parameters(self):
        raise NotImplementedError()

    @abstractmethod
    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        raise NotImplementedError()

    @abstractmethod
    def get_covariance_matrix(self) -> torch.Tensor:
        raise NotImplementedError()
