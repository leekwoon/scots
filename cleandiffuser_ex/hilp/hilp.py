import numpy as np
from typing import Sequence, Dict, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: Optional[int] = None,
                 activations: Callable = F.gelu,
                 activate_final: bool = False,
                 layer_norm: bool = False):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        if output_dim is not None:
            dims.append(output_dim)

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2 or activate_final:
                # Add activation module instance
                if activations == F.gelu:
                    self.layers.append(nn.GELU())
                elif activations == F.relu:
                     self.layers.append(nn.ReLU())
                else:
                     try:
                         self.layers.append(activations())
                     except:
                         raise ValueError(f"Unsupported activation function: {activations}")

                if layer_norm:
                    self.layers.append(nn.LayerNorm(dims[i+1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class LayerNormRepresentation(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: int, # Final output dimension
                 activate_final: bool = True,
                 use_layer_norm: bool = True,
                 ensemble: bool = True):
        super().__init__()
        self.ensemble = ensemble

        mlp_params = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "activate_final": activate_final,
            "layer_norm": use_layer_norm,
            "activations": F.gelu # Keep consistent with JAX default
        }

        if self.ensemble:
            self.mlp1 = MLP(**mlp_params)
            self.mlp2 = MLP(**mlp_params)
        else:
            self.mlp = MLP(**mlp_params)

    def forward(self, observations: torch.Tensor):
        if self.ensemble:
            out1 = self.mlp1(observations)
            out2 = self.mlp2(observations)
            return out1, out2
        else:
            return self.mlp(observations)


class GoalConditionedPhiValue(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 hidden_dims: Sequence[int],
                 skill_dim: int,
                 use_layer_norm: bool = True,
                 ensemble: bool = True):
        super().__init__()
        self.ensemble = ensemble
        self.skill_dim = skill_dim

        self.phi_net = LayerNormRepresentation(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=skill_dim, # Phi output dim
            activate_final=False, # JAX version had activate_final=False for the phi output layer
            use_layer_norm=use_layer_norm,
            ensemble=ensemble
        )

    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        """Returns the phi representation, taking the first element if ensembled."""
        phi_output = self.phi_net(observations)
        # Return first element of ensemble, consistent with JAX implementation's apparent use
        return phi_output[0] if self.ensemble else phi_output

    def forward(self,
                observations: torch.Tensor,
                goals: torch.Tensor):
        """Calculates value V(s,g) = -||phi(s) - phi(g)||."""
        phi_s = self.phi_net(observations)
        phi_g = self.phi_net(goals)

        if self.ensemble:
            dist_sq1 = ((phi_s[0] - phi_g[0])**2).sum(dim=-1)
            dist_sq2 = ((phi_s[1] - phi_g[1])**2).sum(dim=-1)
            v1 = -torch.sqrt(torch.clamp(dist_sq1, min=1e-6))
            v2 = -torch.sqrt(torch.clamp(dist_sq2, min=1e-6))
            return v1, v2
        else:
            dist_sq = ((phi_s - phi_g)**2).sum(dim=-1)
            v = -torch.sqrt(torch.clamp(dist_sq, min=1e-6))
            return v


class HILP(nn.Module):
    def __init__(
        self, 
        obs_dim, 
        skill_dim,
        device,
        value_hidden_dims=(512, 512, 512),
        use_layer_norm=True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.value_hidden_dims = value_hidden_dims
        self.use_layer_norm = use_layer_norm
        self.device = device

        self.value = GoalConditionedPhiValue(
            obs_dim=obs_dim,
            hidden_dims=self.value_hidden_dims,
            skill_dim=self.skill_dim,
            use_layer_norm=self.use_layer_norm,
            ensemble=True # HILP uses ensemble V based on phi distance
        ).to(device)
        self.target_value = GoalConditionedPhiValue(
             obs_dim=obs_dim,
            hidden_dims=self.value_hidden_dims,
            skill_dim=self.skill_dim,
            use_layer_norm=self.use_layer_norm,
            ensemble=True
        ).to(device)
        self.target_value.load_state_dict(self.value.state_dict())
        for p in self.target_value.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_phi(self, observations: np.ndarray) -> np.ndarray:
        """Computes the phi representation (using the first ensemble member)."""
        self.eval() # Set to evaluation mode
        obs_tensor = torch.from_numpy(observations).to(self.device).float()

        is_batch = len(obs_tensor.shape) > 1
        if not is_batch:
             obs_tensor = obs_tensor.unsqueeze(0)

        phi_tensor = self.value.get_phi(obs_tensor)

        if not is_batch:
            phi_tensor = phi_tensor.squeeze(0)

        self.train() # Set back to training mode
        return phi_tensor.cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, self.device))