from typing import Sequence, Optional, Callable, Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import TransformedDistribution, Independent
from torch.distributions.normal import Normal
from torch.distributions.transforms import TanhTransform


class MLP_torch(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: Optional[int] = None,
                 activations: Callable = nn.GELU,
                 activate_final: bool = False,
                 layer_norm: bool = False,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        if output_dim is not None:
            dims.append(output_dim)

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2 or (i == len(dims) - 2 and activate_final):
                self.layers.append(activations())
                if layer_norm:
                    self.layers.append(nn.LayerNorm(dims[i+1], eps=layer_norm_eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layers:
            x = layer_module(x)
        return x

class GCValue_torch(nn.Module):
    def __init__(self,
                 mlp_input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: int = 1,
                 use_layer_norm: bool = True,
                 layer_norm_eps: float = 1e-5,
                 ensemble: bool = True,
                 activations: Callable = nn.GELU): # GCIQL JAX uses default activations
        super().__init__()
        self.ensemble = ensemble
        self.output_dim = output_dim

        mlp_params = {
            "input_dim": mlp_input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": self.output_dim,
            "activate_final": False,
            "layer_norm": use_layer_norm,
            "layer_norm_eps": layer_norm_eps,
            "activations": activations
        }

        if self.ensemble:
            self.mlp1 = MLP_torch(**mlp_params)
            self.mlp2 = MLP_torch(**mlp_params)
        else:
            self.mlp = MLP_torch(**mlp_params)

    def forward(self, processed_mlp_input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.ensemble:
            v1 = self.mlp1(processed_mlp_input)
            v2 = self.mlp2(processed_mlp_input)
            if self.output_dim == 1: # Typical for Q-values
                return v1.squeeze(-1), v2.squeeze(-1)
            return v1, v2
        else:
            v = self.mlp(processed_mlp_input)
            if self.output_dim == 1: # Typical for V-values
                return (v.squeeze(-1),)
            return (v,)

class GCActor_torch(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 hidden_dims: Sequence[int],
                 use_layer_norm: bool = True, # GCIQL JAX GCActor's internal MLP may use it
                 layer_norm_eps: float = 1e-5,
                 const_std: bool = True,
                 state_dependent_std: bool = False,
                 log_std_min: Optional[float] = -5.0,
                 log_std_max: Optional[float] = 2.0,
                 tanh_squash: bool = True,
                 final_fc_init_scale: float = 1e-2,
                 activations: Callable = nn.GELU): # GCIQL JAX uses default activations
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.const_std_is_true_implies_log_std_one = const_std
        self.state_dependent_std = state_dependent_std
        self.tanh_squash, self.action_dim = tanh_squash, action_dim
        self.use_learnable_log_std_param = (not state_dependent_std) and (not self.const_std_is_true_implies_log_std_one)

        current_input_dim = obs_dim + goal_dim
        self.actor_net = MLP_torch(
            input_dim=current_input_dim, hidden_dims=hidden_dims, output_dim=None,
            activate_final=True, layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps, activations=activations
        )
        actor_net_output_dim = hidden_dims[-1] if hidden_dims else current_input_dim
        self.mean_net = nn.Linear(actor_net_output_dim, action_dim)

        if self.state_dependent_std:
            self.log_std_net = nn.Linear(actor_net_output_dim, action_dim)
        elif self.use_learnable_log_std_param:
            self.log_stds_param = nn.Parameter(torch.zeros(action_dim)) # JAX GCActor log_std_param init with zeros

        with torch.no_grad():
            self.mean_net.weight.mul_(final_fc_init_scale)
            self.mean_net.bias.mul_(final_fc_init_scale)
            if hasattr(self, 'log_std_net'): # Should not be hit for GCIQL
                self.log_std_net.weight.mul_(final_fc_init_scale)
                self.log_std_net.bias.mul_(final_fc_init_scale)

    def forward(self,
                observations: torch.Tensor,
                goals: torch.Tensor,
                temperature: float = 1.0) -> torch.distributions.Distribution:
        if temperature <= 0:
             raise ValueError("Temperature for GCActor_torch.forward must be positive.")

        features = self.actor_net(torch.cat([observations, goals], dim=-1))
        means = self.mean_net(features)

        log_stds: torch.Tensor
        if self.state_dependent_std: # False for GCIQL
            log_stds = self.log_std_net(features)
        elif self.const_std_is_true_implies_log_std_one: # True for GCIQL -> log_std=1.0
            log_stds = torch.ones_like(means)
        elif self.use_learnable_log_std_param: # JAX const_std=None case
             log_stds = self.log_stds_param.expand_as(means)
        else:
            raise RuntimeError("Invalid std configuration in GCActor_torch")

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        stds = torch.exp(log_stds) * temperature
        
        base_dist = Independent(Normal(means, stds), 1) # Multivariate Normal with diagonal covariance
        return TransformedDistribution(base_dist, TanhTransform(cache_size=1)) if self.tanh_squash else base_dist


class GCIQL(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 goal_dim: Optional[int] = None, # GCIQL JAX: ex_goals = ex_observations
                 actor_hidden_dims: Sequence[int] = (512, 512, 512), # From GCIQL JAX config
                 value_hidden_dims: Sequence[int] = (512, 512, 512), # From GCIQL JAX config
                 layer_norm: bool = True,        # From GCIQL JAX config
                 const_std_actor: bool = True,   # From GCIQL JAX config (config.const_std)
                 device: str = 'cuda'):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim # This is the size of the continuous action vector
        self.goal_dim = goal_dim if goal_dim is not None else self.obs_dim # GCIQL sets ex_goals = ex_observations
        self.device = device
        self.const_std_actor = const_std_actor # Passed to GCActor_torch
        
        default_activation_fn = nn.GELU # Consistent with HIQL modules

        self.value = GCValue_torch(
            mlp_input_dim=self.obs_dim + self.goal_dim,
            hidden_dims=value_hidden_dims,
            output_dim=1,
            use_layer_norm=layer_norm,
            ensemble=False,
            activations=default_activation_fn
        ).to(device)

        self.critic = GCValue_torch(
            mlp_input_dim=self.obs_dim + self.goal_dim + self.action_dim,
            hidden_dims=value_hidden_dims,
            output_dim=1,
            use_layer_norm=layer_norm,
            ensemble=True,
            activations=default_activation_fn
        ).to(device)
        
        self.target_critic = GCValue_torch(
            mlp_input_dim=self.obs_dim + self.goal_dim + self.action_dim,
            hidden_dims=value_hidden_dims,
            output_dim=1,
            use_layer_norm=layer_norm,
            ensemble=True,
            activations=default_activation_fn
        ).to(device)
        
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters(): p.requires_grad = False

        self.actor = GCActor_torch(
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            hidden_dims=actor_hidden_dims,
            use_layer_norm=False,
            const_std=self.const_std_actor,
            state_dependent_std=False, # Explicitly False in JAX GCIQL
            tanh_squash=True, # GCActor default in JAX utils.networks is True
            activations=default_activation_fn
        ).to(device)

    @torch.no_grad()
    def sample_actions(self,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       seed: Optional[int] = None,
                       temperature: float = 1.0 # JAX GCIQL uses this directly
                       ) -> np.ndarray:
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)

        obs_tensor = torch.from_numpy(observations).to(self.device).float()
        goal_tensor = torch.from_numpy(goals).to(self.device).float()
        
        is_single_input = obs_tensor.ndim == 1
        if is_single_input:
            obs_tensor = obs_tensor.unsqueeze(0)
            goal_tensor = goal_tensor.unsqueeze(0)

        actor_temp_for_dist = temperature if temperature > 0.0 else 1e-6
        actor_dist = self.actor(obs_tensor, goal_tensor, temperature=actor_temp_for_dist)
        
        actions: torch.Tensor
        if temperature == 0.0: # Deterministic action
            actions = actor_dist.mean 
            if self.actor.tanh_squash:
                actions = torch.tanh(actions) # Apply tanh if mean is pre-squashing
        else: # Stochastic action
            actions = actor_dist.sample() # Already squashed if tanh_squash=True in actor

        actions_np = actions.cpu().numpy()
        actions_np = np.clip(actions_np, -1.0, 1.0) # JAX GCIQL clips actions

        return actions_np.squeeze(0) if is_single_input else actions_np

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))