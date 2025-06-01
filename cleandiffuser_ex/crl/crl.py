from typing import Sequence, Optional, Callable, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import TransformedDistribution, Independent
from torch.distributions.normal import Normal
from torch.distributions.transforms import TanhTransform
import math # For sqrt


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
        else: 
            if not hidden_dims:
                 output_dim = input_dim # Edge case: MLP with no hidden layers
            else:
                 output_dim = hidden_dims[-1]

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            is_last_hidden_layer = (i == len(dims) - 2)
            apply_activation_and_norm = True
            if is_last_hidden_layer and not activate_final:
                 apply_activation_and_norm = False

            if apply_activation_and_norm:
                self.layers.append(activations())
                if layer_norm:
                    self.layers.append(nn.LayerNorm(dims[i+1], eps=layer_norm_eps))

        if output_dim is not None and len(dims) > 1 and dims[-1] == output_dim:
             if activate_final:
                 self.layers.append(activations())
                 if layer_norm:
                      self.layers.append(nn.LayerNorm(output_dim, eps=layer_norm_eps))
        elif output_dim is not None and (len(dims) == 1 or dims[-1] != output_dim):
             last_dim_before_output = dims[-1] if len(dims) > 1 else input_dim
             self.layers.append(nn.Linear(last_dim_before_output, output_dim))
             if activate_final:
                  self.layers.append(activations())
                  if layer_norm:
                       self.layers.append(nn.LayerNorm(output_dim, eps=layer_norm_eps))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layers:
            x = layer_module(x)
        return x


class GCActor_torch(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 hidden_dims: Sequence[int],
                 use_layer_norm: bool = False, # Typically False for Actor MLP in these configs
                 layer_norm_eps: float = 1e-5,
                 const_std: bool = True,
                 state_dependent_std: bool = False, # False for CRL/GCIQL
                 log_std_min: Optional[float] = -5.0,
                 log_std_max: Optional[float] = 2.0,
                 tanh_squash: bool = True, # True for CRL/GCIQL Actor
                 final_fc_init_scale: float = 1e-2, # JAX default_init scale=1e-2
                 activations: Callable = nn.GELU):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.const_std = const_std
        self.state_dependent_std = state_dependent_std
        self.tanh_squash = tanh_squash
        self.action_dim = action_dim
        self.use_learnable_log_std_param = (not state_dependent_std) and (not const_std)

        current_input_dim = obs_dim + goal_dim
        # Actor's internal MLP ('actor_net' in JAX)
        self.actor_net = MLP_torch(
            input_dim=current_input_dim,
            hidden_dims=hidden_dims,
            output_dim=None, # Output dim is the last hidden dim
            activate_final=True, # JAX GCActor MLP activates final layer
            layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            activations=activations
        )
        actor_net_output_dim = hidden_dims[-1] if hidden_dims else current_input_dim

        self.mean_net = nn.Linear(actor_net_output_dim, action_dim)

        if self.state_dependent_std: # False for CRL/GCIQL
            self.log_std_net = nn.Linear(actor_net_output_dim, action_dim)
        elif self.use_learnable_log_std_param: # Only if const_std=False
            self.log_stds_param = nn.Parameter(torch.zeros(action_dim)) # JAX GCActor log_stds init with zeros

        with torch.no_grad():
            nn.init.xavier_uniform_(self.mean_net.weight, gain=final_fc_init_scale)
            nn.init.constant_(self.mean_net.bias, 0.0) # Or scale bias too if needed: self.mean_net.bias.mul_(final_fc_init_scale)

            if hasattr(self, 'log_std_net'): # False for CRL/GCIQL
                nn.init.xavier_uniform_(self.log_std_net.weight, gain=final_fc_init_scale)
                nn.init.constant_(self.log_std_net.bias, 0.0)

    def forward(self,
                observations: torch.Tensor,
                goals: torch.Tensor,
                temperature: float = 1.0) -> torch.distributions.Distribution:
        if temperature <= 0:
             raise ValueError("Temperature for GCActor_torch.forward must be positive.")

        inputs = torch.cat([observations, goals], dim=-1)
        features = self.actor_net(inputs)
        means = self.mean_net(features)

        log_stds: torch.Tensor
        if self.state_dependent_std: # False for CRL/GCIQL
            log_stds = self.log_std_net(features)
        elif self.const_std: # True for CRL/GCIQL -> log_std=0.0 => std=1.0
            log_stds = torch.zeros_like(means)
        elif self.use_learnable_log_std_param: # Only if const_std=False
             log_stds = self.log_stds_param.expand_as(means)
        else:
            raise RuntimeError("Invalid std configuration in GCActor_torch")

        if self.log_std_min is not None and self.log_std_max is not None:
            log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)

        stds = torch.exp(log_stds) * temperature
        base_dist = Independent(Normal(loc=means, scale=stds), 1) # Corresponds to MultivariateNormalDiag

        if self.tanh_squash:
            return TransformedDistribution(base_dist, TanhTransform(cache_size=1))
        else:
            return base_dist


class GCBilinearValue_torch(nn.Module):
    """Goal-conditioned bilinear value/critic function (PyTorch)."""
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 action_dim: Optional[int], # None for V-function, int for Q-function
                 hidden_dims: Sequence[int],
                 latent_dim: int,
                 use_layer_norm: bool = True,
                 layer_norm_eps: float = 1e-5,
                 ensemble: bool = True,
                 value_exp: bool = False, # CRL uses value_exp=True
                 activations: Callable = nn.GELU,
                 ):
        super().__init__()
        self.ensemble = ensemble
        self.value_exp = value_exp
        self.latent_dim = latent_dim
        self.action_dim = action_dim # Store action_dim to determine phi input size

        phi_input_dim = obs_dim
        if self.action_dim is not None: # If it's a Q-function (critic)
            phi_input_dim += action_dim
        psi_input_dim = goal_dim

        mlp_params = {
            "hidden_dims": hidden_dims,
            "output_dim": latent_dim, # phi and psi output latent vectors
            "activate_final": False, # JAX GCBilinearValue MLP activate_final=False
            "layer_norm": use_layer_norm,
            "layer_norm_eps": layer_norm_eps,
            "activations": activations
        }

        if self.ensemble:
            self.phi1 = MLP_torch(input_dim=phi_input_dim, **mlp_params)
            self.psi1 = MLP_torch(input_dim=psi_input_dim, **mlp_params)
            self.phi2 = MLP_torch(input_dim=phi_input_dim, **mlp_params)
            self.psi2 = MLP_torch(input_dim=psi_input_dim, **mlp_params)
        else:
            self.phi = MLP_torch(input_dim=phi_input_dim, **mlp_params)
            self.psi = MLP_torch(input_dim=psi_input_dim, **mlp_params)

    def _compute_bilinear_value(self, phi_out, psi_out):
        """Computes scaled dot product and optionally exponentiates."""
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive for GCBilinearValue")
        v = (phi_out * psi_out).sum(dim=-1) / math.sqrt(self.latent_dim)
        if self.value_exp:
            v = torch.exp(v)
        return v

    def forward(self,
                observations: torch.Tensor,
                goals: torch.Tensor,
                actions: Optional[torch.Tensor] = None,
                info: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Return the value/critic function.

        Args:
            observations: Observations tensor.
            goals: Goals tensor.
            actions: Actions tensor (optional, required for Q-critic).
            info: Whether to additionally return the representations phi and psi.

        Returns:
            If ensemble=False:
                if info=False: v (Tensor)
                if info=True: (v, phi_out, psi_out) (Tuple[Tensor, Tensor, Tensor])
            If ensemble=True:
                if info=False: (v1, v2) (Tuple[Tensor, Tensor])
                if info=True: (v1, v2, phi1_out, phi2_out, psi1_out, psi2_out) (Tuple[Tensor, ...])
        """
        obs_encoded = observations
        goal_encoded = goals

        if self.action_dim is not None: # Q-function
            if actions is None:
                raise ValueError("Actions must be provided for GCBilinearValue critic (action_dim is not None).")
            if actions.shape[:-1] != obs_encoded.shape[:-1]:
                 raise ValueError(f"Action batch shape {actions.shape[:-1]} doesn't match observation batch shape {obs_encoded.shape[:-1]}")
            phi_input = torch.cat([obs_encoded, actions], dim=-1)
        else: # V-function
            if actions is not None:
                 print("Warning: Actions provided to GCBilinearValue V-function (action_dim=None), they will be ignored.")
            phi_input = obs_encoded

        if self.ensemble:
            phi1_out = self.phi1(phi_input)
            psi1_out = self.psi1(goal_encoded)
            v1 = self._compute_bilinear_value(phi1_out, psi1_out)

            phi2_out = self.phi2(phi_input)
            psi2_out = self.psi2(goal_encoded)
            v2 = self._compute_bilinear_value(phi2_out, psi2_out)

            if info:
                return v1, v2, phi1_out, phi2_out, psi1_out, psi2_out
            else:
                return v1, v2
        else: # Not an ensemble
            phi_out = self.phi(phi_input)
            psi_out = self.psi(goal_encoded)
            v = self._compute_bilinear_value(phi_out, psi_out)

            if info:
                return v, phi_out, psi_out
            else:
                return v


class CRL(nn.Module):
    """Contrastive RL (CRL) agent (PyTorch). Continuous actions only."""
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 goal_dim: Optional[int] = None,
                 actor_hidden_dims: Sequence[int] = (512, 512, 512), # Default from CRL JAX config
                 value_hidden_dims: Sequence[int] = (512, 512, 512), # Default from CRL JAX config
                 latent_dim: int = 512,            # Default from CRL JAX config
                 layer_norm: bool = True,        # Default from CRL JAX config (applies to Value/Critic MLPs)
                 const_std_actor: bool = True,   # Default from CRL JAX config
                 actor_loss: str = 'ddpgbc',     # Default from CRL JAX config ('awr' or 'ddpgbc')
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        if actor_loss not in ['awr', 'ddpgbc']:
            raise ValueError(f"Unsupported actor_loss: {actor_loss}. Must be 'awr' or 'ddpgbc'.")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim if goal_dim is not None else self.obs_dim # JAX CRL sets ex_goals = ex_observations
        self.device = device
        self.actor_loss_type = actor_loss
        self.const_std_actor = const_std_actor # Passed to GCActor_torch

        default_activation_fn = nn.GELU

        self.critic = GCBilinearValue_torch(
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim, # Critic takes actions
            hidden_dims=value_hidden_dims,
            latent_dim=latent_dim,
            use_layer_norm=layer_norm,
            ensemble=True, # CRL Critic is ensemble
            value_exp=True, # CRL uses value_exp=True
            activations=default_activation_fn
        ).to(device)

        self.value = None
        if self.actor_loss_type == 'awr':
            self.value = GCBilinearValue_torch(
                obs_dim=self.obs_dim,
                goal_dim=self.goal_dim,
                action_dim=None, # Value function does not take actions
                hidden_dims=value_hidden_dims,
                latent_dim=latent_dim,
                use_layer_norm=layer_norm,
                ensemble=False, # CRL Value is NOT ensemble
                value_exp=True, # CRL uses value_exp=True
                activations=default_activation_fn
            ).to(device)

        self.actor = GCActor_torch(
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            hidden_dims=actor_hidden_dims,
            use_layer_norm=False, # Actor MLP layer_norm usually off in these configs
            const_std=self.const_std_actor,
            state_dependent_std=False, # Explicitly False in JAX CRL
            tanh_squash=True, # GCActor default is True
            activations=default_activation_fn
        ).to(device)

    @torch.no_grad()
    def sample_actions(self,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       seed: Optional[int] = None,
                       temperature: float = 1.0 # JAX CRL uses this directly
                       ) -> np.ndarray:
        """Samples actions from the actor. Clips to [-1, 1]."""
        self.eval() # Set to evaluation mode

        if seed is not None:
            torch.manual_seed(seed)
            if self.device == 'cuda': torch.cuda.manual_seed_all(seed)

        obs_tensor = torch.from_numpy(observations).to(self.device).float()
        goal_tensor = torch.from_numpy(goals).to(self.device).float()

        is_single_input = obs_tensor.ndim == 1
        if is_single_input:
            obs_tensor = obs_tensor.unsqueeze(0)
            goal_tensor = goal_tensor.unsqueeze(0)

        actor_temp_for_dist = temperature if temperature > 0.0 else 1e-6 # Use small epsilon if temp is 0
        actor_dist = self.actor(obs_tensor, goal_tensor, temperature=actor_temp_for_dist)

        actions: torch.Tensor
        if temperature == 0.0: # Deterministic action (mean/mode)
            actions = actor_dist.mean
            if self.actor.tanh_squash:
                if isinstance(actor_dist, TransformedDistribution):
                    for transform in actor_dist.transforms:
                         if isinstance(transform, TanhTransform):
                              actions = transform(actions)
                              break
                else: actions = torch.tanh(actions)

        else: # Stochastic action
            actions = actor_dist.sample() # sample() handles transforms automatically

        actions_np = actions.cpu().numpy()
        actions_np = np.clip(actions_np, -1.0, 1.0)

        return actions_np.squeeze(0) if is_single_input else actions_np

    def save(self, path: str):
        """Saves the agent's state dictionary."""
        print(f"Saving CRLAgent_torch state_dict to: {path}")
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Loads the agent's state dictionary."""
        print(f"Loading CRLAgent_torch state_dict from: {path}")
        self.load_state_dict(torch.load(path, map_location=self.device))