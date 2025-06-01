import math
import os
import random
from typing import Union, Dict, Callable

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """ Set seed for reproducibility """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def at_least_ndim(x: Union[np.ndarray, torch.Tensor, int, float], ndim: int, pad: int = 0):
    """ Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (np.ndarray, list, tuple, int, float)):
        return torch.tensor(x, device=device)
    else:
        raise ValueError(f"Unsupported type {type(x)}")


# ================ Discretization ==================
def uniform_discretization(T: int = 1000, eps: float = 1e-3):
    return torch.linspace(eps, 1.0, T)


SUPPORTED_DISCRETIZATIONS = {
    "uniform": uniform_discretization,
}


# ================= Noise schedules =================
def linear_noise_schedule(
    t_diffusion: torch.Tensor, beta0: float = 0.1, beta1: float = 20.0
):
    log_alpha = -(beta1 - beta0) / 4.0 * (t_diffusion**2) - beta0 / 2.0 * t_diffusion
    alpha = log_alpha.exp()
    sigma = (1.0 - alpha**2).sqrt()
    return alpha, sigma


def inverse_linear_noise_schedule(
    alpha: torch.Tensor = None,
    sigma: torch.Tensor = None,
    logSNR: torch.Tensor = None,
    beta0: float = 0.1,
    beta1: float = 20.0,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = (alpha / sigma).log() if logSNR is None else logSNR
    t_diffusion = (2 * (1 + (-2 * lmbda).exp()).log() /
                   (beta0 + (beta0**2 + 2 * (beta1 - beta0) * (1 + (-2 * lmbda).exp()).log())))
    return t_diffusion


def cosine_noise_schedule(t_diffusion: torch.Tensor, s: float = 0.008):
    alpha = (np.pi / 2.0 * ((t_diffusion).clip(0., 0.9946) + s) / (1 + s)).cos() / np.cos(
        np.pi / 2.0 * s / (1 + s))
    sigma = (1.0 - alpha**2).sqrt()
    return alpha, sigma


def inverse_cosine_noise_schedule(
    alpha: torch.Tensor = None,
    sigma: torch.Tensor = None,
    logSNR: torch.Tensor = None,
    s: float = 0.008,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = (alpha / sigma).log() if logSNR is None else logSNR
    t_diffusion = (
        2 * (1 + s) / np.pi * torch.arccos((
            -0.5 * (1 + (-2 * lmbda).exp()).log()
            + np.log(np.cos(np.pi * s / 2 / (s + 1)))).exp()) - s)
    return t_diffusion


SUPPORTED_NOISE_SCHEDULES = {
    "linear": {
        "forward": linear_noise_schedule,
        "reverse": inverse_linear_noise_schedule,
    },
    "cosine": {
        "forward": cosine_noise_schedule,
        "reverse": inverse_cosine_noise_schedule,
    },
}

# ================= Sampling step schedule ===============
def uniform_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10):
    return torch.linspace(0, T - 1, sampling_steps + 1, dtype=torch.long)


def uniform_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10):
    if trange is None:
        trange = [1e-3, 1.0]
    return torch.linspace(trange[0], trange[1], sampling_steps + 1, dtype=torch.float32)


SUPPORTED_SAMPLING_STEP_SCHEDULE = {
    "uniform": uniform_sampling_step_schedule,
    "uniform_continuous": uniform_sampling_step_schedule_continuous,
}


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ema_update(model: nn.Module, model_ema: nn.Module, ema_rate: float):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data.mul_(ema_rate).add_(param.data, alpha=1 - ema_rate)


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures,
# from https://github.com/NVlabs/edm/blob/main/training/networks.py#L269
class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# -----------------------------------------------------------
# Timestep embedding used in Transformer
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.einsum('...i,j->...ij', x, emb.to(x.dtype))
        # emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures
class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor):
        emb = torch.einsum('...i,j->...ij', x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return self.mlp(emb)
    

SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding,
}


# -----------------------------------------------------------
# Beautiful model size visualization from https://github.com/jannerm/diffuser/tree/main


def _to_str(num):
    if num >= 1e6:
        return f"{(num / 1e6):.2f} M"
    else:
        return f"{(num / 1e3):.2f} k"


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters() if p.requires_grad}
    n_parameters = sum(counts.values())
    print(f"Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        f"... and {len(counts) - topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
    return n_parameters


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result
    

def loop_dataloader(dl):
    while True:
        for b in dl:
            yield b
