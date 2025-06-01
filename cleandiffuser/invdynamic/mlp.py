import torch
import torch.nn as nn

from cleandiffuser.utils import Mlp


class MlpInvDynamic:
    """ Simple MLP-based inverse dynamics model. The model is a 3-layer MLP with ReLU activation.

    Args:
        o_dim: int,
            Dimension of observation space.
        a_dim: int,
            Dimension of action space.
        hidden_dim: int,
            Dimension of hidden layers. Default: 512.
        out_activation: nn.Module,
            Activation function for output layer. Default: nn.Tanh().
        optim_params: dict,
            Optimizer parameters. Default: {}.
        device: str,
            Device for the model. Default: "cpu".

    Examples:
        >>> invdyn = MlpInvDynamic(3, 2)
        >>> invdyn.train()
        >>> batch = ...
        >>> obs, act, obs_next = batch
        >>> loss = invdyn.update(obs, act, obs_next)
        >>> invdyn.eval()
        >>> pred_act = invdyn.predict(obs, obs_next)
    """
    def __init__(
            self,
            o_dim: int,
            a_dim: int,
            hidden_dim: int = 512,
            out_activation: nn.Module = nn.Tanh(),
            optim_params: dict = {},
            device: str = "cpu",
    ):
        self.device = device
        self.o_dim, self.a_dim, self.hidden_dim = o_dim, a_dim, hidden_dim
        self.out_activation = out_activation
        self.optim_params = optim_params
        params = {"lr": 5e-4}
        params.update(optim_params)
        self.mlp = Mlp(
            2 * o_dim, [hidden_dim, hidden_dim], a_dim,
            nn.ReLU(), out_activation).to(device)
        self.optim = torch.optim.Adam(self.mlp.parameters(), **optim_params)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, o, o_next):
        return self.mlp(torch.cat([o, o_next], dim=-1))

    def update(self, o, a, o_next):
        self.optim.zero_grad()
        a_pred = self.forward(o, o_next)
        loss = ((a_pred - a) ** 2).mean()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def predict(self, o, o_next):
        return self.forward(o, o_next)

    def __call__(self, o, o_next):
        return self.predict(o, o_next)

    def train(self):
        self.mlp.train()

    def eval(self):
        self.mlp.eval()

    def save(self, path):
        torch.save(self.mlp.state_dict(), path)

    def load(self, path):
        self.mlp.load_state_dict(torch.load(path, self.device))


class FancyMlpInvDynamic:
    """ Fancy MLP-based inverse dynamics model. The model is a 3-layer MLP with GELU activation. It also includes
    optional LayerNorm and Dropout. We suggest using 0.1 Dropout and LayerNorm for better performance.

    Args:
        o_dim: int,
            Dimension of observation space.
        a_dim: int,
            Dimension of action space.
        hidden_dim: int,
            Dimension of hidden layers. Default: 256.
        out_activation: nn.Module,
            Activation function for output layer. Default: nn.Tanh().
        add_norm: bool,
            Whether to add LayerNorm. Default: False.
        add_dropout: bool,
            Whether to add Dropout. Default: False.
        optim_params: dict,
            Optimizer parameters. Default: {}.
        device: str,
            Device for the model. Default: "cpu".

    Examples:
        >>> invdyn = FancyMlpInvDynamic(3, 2, add_norm=True, add_dropout=True)
        >>> invdyn.train()
        >>> batch = ...
        >>> obs, act, obs_next = batch
        >>> loss = invdyn.update(obs, act, obs_next)
        >>> invdyn.eval()
        >>> pred_act = invdyn.predict(obs, obs_next)
    """
    def __init__(
            self, o_dim: int, a_dim: int, hidden_dim: int = 256,
            out_activation: nn.Module = nn.Tanh(),
            add_norm: bool = False, add_dropout: bool = False,
            optim_params: dict = {}, device: str = "cpu",
    ):
        self.device = device
        self.o_dim, self.a_dim, self.hidden_dim = o_dim, a_dim, hidden_dim
        self.out_activation = out_activation
        self.optim_params = optim_params
        params = {"lr": 3e-4}
        params.update(optim_params)

        self.model = nn.Sequential(
            nn.Linear(2 * o_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim) if add_norm else nn.Identity(),
            nn.Dropout(0.1) if add_dropout else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, a_dim), out_activation).to(device)

        self.optim = torch.optim.Adam(self.model.parameters(), **optim_params)

    def forward(self, o, o_next):
        return self.model(torch.cat([o, o_next], dim=-1))

    def update(self, o, a, o_next):
        self.optim.zero_grad()
        a_pred = self.forward(o, o_next)
        loss = ((a_pred - a) ** 2).mean()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def predict(self, o, o_next):
        return self.forward(o, o_next)

    def __call__(self, o, o_next):
        return self.predict(o, o_next)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, self.device))