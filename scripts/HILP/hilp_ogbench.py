# -------------------------
import os
import ogbench
import glob
import pickle
import tqdm
import numpy as np
import pyrallis
import random
import os
import wandb
from datetime import datetime
import uuid
import copy
import tempfile
import functools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from sklearn.manifold import TSNE

import dataclasses
from typing import Sequence, Dict, Any, Mapping, Callable # Use standard typing
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
import optax


nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


@dataclasses.dataclass
class Config:
    # core config
    lr = 3e-4
    value_hidden_dims: Sequence[int] = (512, 512, 512)
    discount: float = 0.99
    tau: float = 0.005         # Target network update rate
    expectile: float = 0.95    # Expectile for value loss
    use_layer_norm: int = 1  # 1 for True, 0 for False
    skill_dim: int = 32      # Dimension of the phi representation

    # dataset config
    env_name: str = "pointmaze-large-stitch-v0"  # OpenAI gym environment name
    batch_size: int = 1024
    p_currgoal: float = 0.0
    p_trajgoal: float = 0.625
    p_randomgoal: float = 0.375
    geom_sample: int = 1     # 1 for True (geometric), 0 for False (uniform)

    # train config
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    train_steps: int = 1000000
    save_dir: str = 'exp/'
    log_interval: int = 1000
    viz_interval: int = 5000
    save_interval: int = 100000 # Save checkpoints every N steps

    # wandb config
    project: str = "scots-ogbench" # Changed project name
    group: str = "debug"
    name: str = "HILP_jax"

    def __post_init__(self):
        self.run_name = f"{self.name}-{self.env_name}"
        self.run_name += f'-{datetime.now().strftime("%Y%m%d_%H%M%S")}'

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    discount: float
    geom_sample: int = 1
    terminal_key: str = 'dones_float'

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)

        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)

        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)

        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None, evaluation=False):
        if indx is None:
            indx = np.random.randint(self.dataset.size - 1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)

        batch['rewards'] = success.astype(float) - 1.0
        batch['masks'] = (1.0 - success.astype(float))
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        return batch


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info
    
def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Callable = functools.partial(nn.gelu, approximate=False) # Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm(epsilon=1e-5)(x)
        return x
    

class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    use_layer_norm: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        # module = LayerNormMLP
        # if self.ensemble:
            # module = ensemblize(module, 2)
        return mlp_module(self.hidden_dims, activate_final=self.activate_final, layer_norm=self.use_layer_norm)(observations)


class GoalConditionedPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,) # This seems unused? Kept for compatibility.
    skill_dim: int = 2 # This is the output dim of phi
    use_layer_norm: bool = True
    ensemble: bool = True

    def setup(self) -> None:
        repr_class = LayerNormRepresentation 
        # The phi network IS the value network here
        phi = repr_class((*self.hidden_dims, self.skill_dim), activate_final=False, use_layer_norm=self.use_layer_norm, ensemble=self.ensemble)
        self.phi = phi

    def get_phi(self, observations):
        # Ensure it returns the phi representation, potentially averaging over the ensemble if needed.
        # HILP original uses the first element of the ensemble.
        phi_output = self.phi(observations)
        return phi_output[0] if self.ensemble else phi_output

    def __call__(self, observations, goals=None, info=False):
        # Calculates value based on phi distance
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        # If ensemble, phi_s and phi_g will have shape (2, batch, dim)
        # We need distance for each element of the ensemble
        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1) # Shape (2, batch) if ensemble, (batch,) otherwise
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
        return v
    

class HILP(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def value_loss(self, batch, network_params):
        (next_v1, next_v2) = self.network.select('target_value')(batch['next_observations'], batch['goals'])
        next_v = jnp.minimum(next_v1, next_v2)
        # The reward is now directly from the goal-conditioned dataset (0 for not success, scale+shift for success)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2
        (v1, v2) = self.network.select('value')(batch['observations'], batch['goals'], params=network_params)
        v = (v1 + v2) / 2

        value_loss1 = expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        # Metrics related to the value function V(s, g) = -||phi(s) - phi(g)||
        return value_loss, {
            'value_loss': value_loss,
            'v max': v.max(), # Max -distance
            'v min': v.min(), # Min -distance (should be close to 0)
            'v mean': v.mean(), # Avg -distance
            'abs adv mean': jnp.abs(adv).mean(),
            'adv mean': adv.mean(),
            'adv max': adv.max(),
            'adv min': adv.min(),
            'accept prob': (adv >= 0).mean(),
            'rewards_mean': batch['rewards'].mean(), # Check goal-reaching reward mean
            'masks_mean': batch['masks'].mean(),     # Check goal-reaching mask mean
        }

    @jax.jit
    def total_loss(self, batch, grad_params):
        """Compute the total loss."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        loss = value_loss 
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network), info

    @jax.jit
    def get_phi(self, observations: jnp.ndarray) -> jnp.ndarray:
        phi_module = self.network.model_def.modules['value']
        phi_params = self.network.params['modules_value']
        return phi_module.apply({'params': phi_params}, observations, method=phi_module.get_phi)

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        lr: float = 3e-4,
        value_hidden_dims: Sequence[int] = (512, 512, 512),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.95,
        use_layer_norm: int = 1,
        skill_dim: int = 4, # 32,
        **kwargs):

        print("Creating HILPRepAgent. Extra kwargs:", kwargs)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_goals = ex_observations

        value_def = GoalConditionedPhiValue(
            hidden_dims=value_hidden_dims,
            use_layer_norm=use_layer_norm > 0,
            ensemble=True,
            skill_dim=skill_dim,
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=lr)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**dict(
            discount=discount,
            tau=tau,
            target_update_rate=tau,
            expectile=expectile,
            skill_dim=skill_dim,
        )))


def get_canvas_image(canvas: FigureCanvas) -> np.ndarray:
    """
    Converts a Matplotlib FigureCanvasAgg object to a NumPy array.
    """
    canvas.draw()
    s = canvas.tostring_rgb()
    width, height = canvas.get_width_height()
    image = np.frombuffer(s, dtype=np.uint8).reshape((height, width, 3))
    return image


def generate_tsne_visualization(agent, dataset, env, seed, step_count):

    maze_map = env.unwrapped.maze_map == 1

    UNIT            = float(env.unwrapped._maze_unit)   # = 4.0
    ORIGIN_X        = -float(env.unwrapped._offset_x)   # = -4
    ORIGIN_Y        = -float(env.unwrapped._offset_y)   # = -4
    MAZE_SIZE_SCALING = UNIT       
    WALL_HALF_WIDTH   = UNIT / 2    

    wall_polygons = []
    num_rows = len(maze_map)
    num_cols = len(maze_map[0]) if num_rows > 0 else 0

    for r in range(num_rows):
        for c in range(num_cols):
            if maze_map[r][c] == 1: # Wall
                center_x = ORIGIN_X + c * UNIT
                center_y = ORIGIN_Y + r * UNIT
                min_x, max_x = center_x - WALL_HALF_WIDTH, center_x + WALL_HALF_WIDTH
                min_y, max_y = center_y - WALL_HALF_WIDTH, center_y + WALL_HALF_WIDTH
                wall_polygons.append(box(min_x, min_y, max_x, max_y))

    if not wall_polygons:
         print("Warning: No wall polygons were generated. Check maze map and scaling.")
         return np.zeros((400, 400, 3), dtype=np.uint8) 

    poly_union = unary_union(wall_polygons)
    print(f"Built wall union from {len(wall_polygons)} polygons.")

    # --- Sample Traversable Points & Create Observations ---
    obs_list = []
    traversable_points = []
    sampling_step = UNIT / 4

    x_min_bound = ORIGIN_X - WALL_HALF_WIDTH
    x_max_bound = ORIGIN_X + num_cols * UNIT - WALL_HALF_WIDTH
    y_min_bound = ORIGIN_Y - WALL_HALF_WIDTH
    y_max_bound = ORIGIN_Y + num_rows * UNIT - WALL_HALF_WIDTH

    x_range = np.arange(x_min_bound + sampling_step/2, x_max_bound - sampling_step/2, sampling_step)
    y_range = np.arange(y_min_bound + sampling_step/2, y_max_bound - sampling_step/2, sampling_step)

    print(f"Checking points in x range: {x_range.min():.1f} to {x_range.max():.1f}, y range: {y_range.min():.1f} to {y_range.max():.1f}")

    example_obs_full_tree = jax.tree_map(lambda x: x[0], dataset['observations'])
    example_obs_flat = list(jax.tree_util.tree_leaves(example_obs_full_tree))[0]
    obs_dim = example_obs_flat.shape[0]
    if obs_dim > 2:
        remaining_obs_template = np.asarray(example_obs_flat[2:]) 
        print(f"Using template for non-coordinate part (dim={remaining_obs_template.shape[0]}) from dataset.")
    else:
        remaining_obs_template = np.array([], dtype=example_obs_flat.dtype)
        print("Observations seem to contain only coordinates.")

    point_count = 0
    traversable_count = 0
    for x in x_range:
        for y in y_range:
            point_count += 1
            point = Point(x, y)
            if not poly_union.intersects(point): 
                traversable_count += 1
                current_obs_coords = np.array([x, y])
                current_obs = np.concatenate([current_obs_coords, remaining_obs_template])
                obs_list.append(current_obs)
                traversable_points.append((x, y))

    print(f"Checked {point_count} points. Found {traversable_count} traversable points.")

    if not obs_list:
         print("Warning: No traversable points found for visualization.")
         return np.zeros((400, 400, 3), dtype=np.uint8) 

    obs_array = np.array(obs_list)
    traversable_points_array = np.array(traversable_points)

    # --- Calculate Hilbert Representations ---
    print(f"Calculating Hilbert representations for {len(obs_array)} points...")
    phi_raw = agent.get_phi(obs_array)
    ex_phis = np.asarray(phi_raw) 
    print(f"Calculated {ex_phis.shape} Hilbert representations.")

    # --- Perform t-SNE and Plot ---
    image = None
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200)
    canvas = FigureCanvas(fig)

    if len(ex_phis) <= 1:
         print("Warning: Need more than 1 point for t-SNE.")
         plt.close(fig)
         return np.zeros((400, 400, 3), dtype=np.uint8) 

    print("Running t-SNE...")
    perplexity_value = min(30.0, float(len(ex_phis) - 1)) 
    if perplexity_value <= 0: 
        perplexity_value = 5.0 
        print(f"Warning: Low sample count ({len(ex_phis)}), setting perplexity to {perplexity_value}")

    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity_value,
                n_iter=300, init='pca', learning_rate='auto', n_jobs=-1)
    try:
        tsne_phis = tsne.fit_transform(ex_phis)
        print("t-SNE finished.")

        # Plot using x coordinate for color
        scatter = ax.scatter(tsne_phis[:, 0], tsne_phis[:, 1], c=traversable_points_array[:, 0], cmap='viridis', s=8, alpha=0.8)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('X coordinate')
        ax.set_title(f't-SNE of Hilbert Representations (Step {step_count}) - Maze2D')
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
        ax.grid(True, linestyle='--', alpha=0.6)

        # Convert plot to image
        plt.tight_layout()
        image = get_canvas_image(canvas)

    except Exception as e:
        print(f"Error during t-SNE or plotting: {e}")
        image = np.zeros((400, 400, 3), dtype=np.uint8) 

    plt.close(fig) 
    return image


@pyrallis.wrap()
def main(config: Config):
    wandb.init(
        config=dataclasses.asdict(config),
        project=config.project,
        group=config.group,
        name=config.run_name,
        id=str(uuid.uuid4()), # Generate unique ID
        dir=tempfile.mkdtemp(), # Use temp dir for wandb files
    )
    os.makedirs(config.save_dir, exist_ok=True)
    print(f"WandB Run Name: {wandb.run.name}")
    print(f"Saving checkpoints and logs to: {config.save_dir}")

    env, dataset, _ = ogbench.make_env_and_datasets(
        config.env_name,
        compact_dataset=False,
    )

    train_dataset = Dataset.create(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32),
        terminals=dataset['terminals'].astype(np.float32),
    )

    train_dataset = GCDataset(
        train_dataset, # Pass the Dataset object from make_env_and_datasets
        p_randomgoal=config.p_randomgoal,
        p_trajgoal=config.p_trajgoal,
        p_currgoal=config.p_currgoal,
        discount=config.discount,
        geom_sample=config.geom_sample,
        terminal_key='terminals',
    )

    # --- Agent Initialization ---
    random.seed(config.seed)
    np.random.seed(config.seed)

    example_batch = train_dataset.sample(1) # Sample a batch for shape inference

    agent = HILP.create(
        seed=config.seed,
        ex_observations=example_batch['observations'],
        lr=config.lr,
        value_hidden_dims=config.value_hidden_dims,
        discount=config.discount,
        tau=config.tau,
        expectile=config.expectile,
        use_layer_norm=config.use_layer_norm,
        skill_dim=config.skill_dim, # Phi output dimension
    )

    # Train agent.
    for i in tqdm.tqdm(range(1, config.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config.batch_size)
        agent, update_info = agent.update(batch)

        # --- Logging ---
        if i % config.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in jax.device_get(update_info).items()} # Ensure metrics are on CPU
            wandb.log(train_metrics, step=i)

        if i % config.viz_interval == 0:
            viz_image = generate_tsne_visualization(
                agent, train_dataset.dataset, env, config.seed, i
            )
            wandb.log({'diagnostics/hilbert_tsne': wandb.Image(viz_image)}, step=i)
            print(f"Step {i}: Logged Hilbert t-SNE visualization to WandB.")
    
        if i % config.save_interval == 0:
            save_agent(agent, config.save_dir, i)

if __name__ == '__main__':
    main()
