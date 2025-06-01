import torch
import torch.nn as nn
import numpy as np
from typing import List


class Mlp(nn.Module):
    """ **Multilayer perceptron.** A simple pytorch MLP module.

    Args:
        in_dim: int,
            The dimension of the input tensor.
        hidden_dims: List[int],
            A list of integers, each element is the dimension of the hidden layer.
        out_dim: int,
            The dimension of the output tensor.
        activation: nn.Module,
            The activation function used in the hidden layers.
        out_activation: nn.Module,
            The activation function used in the output layer.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dims: List[int],
            out_dim: int,
            activation: nn.Module = nn.ReLU(),
            out_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                    activation,
                )
                for i in range(len(hidden_dims))
            ],
            nn.Linear(hidden_dims[-1], out_dim),
            out_activation
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class GroupNorm1d(nn.Module):
    def __init__(self, dim, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, dim // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)
