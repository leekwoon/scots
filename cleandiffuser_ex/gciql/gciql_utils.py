import pickle
import flax.core # For FrozenDict type hint
import flax.serialization
import jax # For tree_map and Array types
import jax.numpy # For Array type
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Sequence, Optional, Callable, Tuple


def convert_flax_dense_to_torch_linear(flax_params: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    torch_params = {}
    if 'kernel' in flax_params:
        kernel_np = np.array(flax_params['kernel'])
        torch_params['weight'] = torch.from_numpy(kernel_np.T).float()
    if 'bias' in flax_params:
        torch_params['bias'] = torch.from_numpy(np.array(flax_params['bias'])).float()
    return torch_params

def convert_flax_layernorm_to_torch_layernorm(flax_params: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    torch_params = {}
    if 'scale' in flax_params:
        torch_params['weight'] = torch.from_numpy(np.array(flax_params['scale'])).float()
    if 'bias' in flax_params:
        torch_params['bias'] = torch.from_numpy(np.array(flax_params['bias'])).float()
    return torch_params

def convert_jax_mlp_to_torch_mlp_modulelist(
    jax_mlp_params: Dict[str, Any],
    torch_mlp_modulelist: nn.ModuleList, # This is the .layers attribute of an MLP_torch instance
    base_key_prefix: str
    ) -> Dict[str, torch.Tensor]:
    torch_state_dict_for_mlp = {}
    flax_layer_idx = 0
    torch_module_idx = 0

    print(f"  [MLP Convert START] JAX keys: {list(jax_mlp_params.keys())}, PyTorch num_modules: {len(torch_mlp_modulelist)}, Prefix: {base_key_prefix}")

    while f'Dense_{flax_layer_idx}' in jax_mlp_params:
        print(f"    Processing JAX Dense_{flax_layer_idx} for PyTorch module index {torch_module_idx}")
        # 1. Expect and load Linear for Dense_{flax_layer_idx}
        if torch_module_idx >= len(torch_mlp_modulelist) or \
           not isinstance(torch_mlp_modulelist[torch_module_idx], nn.Linear):
            raise ValueError(
                f"PyTorch MLP structure mismatch at {base_key_prefix}{torch_module_idx}. "
                f"Expected nn.Linear for JAX Dense_{flax_layer_idx}, "
                f"got {type(torch_mlp_modulelist[torch_module_idx]) if torch_module_idx < len(torch_mlp_modulelist) else 'OutOfBound'}."
            )
        flax_dense_p = jax_mlp_params[f'Dense_{flax_layer_idx}']
        torch_linear_p = convert_flax_dense_to_torch_linear(flax_dense_p)
        for name, tensor in torch_linear_p.items():
            torch_state_dict_for_mlp[f"{base_key_prefix}{torch_module_idx}.{name}"] = tensor
        print(f"      Loaded Linear for Dense_{flax_layer_idx} into PyTorch layer {torch_module_idx}")
        torch_module_idx += 1

        # 2. Skip Activation in torch_mlp_modulelist if present.
        if torch_module_idx < len(torch_mlp_modulelist) and \
           not isinstance(torch_mlp_modulelist[torch_module_idx], (nn.Linear, nn.LayerNorm)):
            print(f"      Skipped PyTorch activation layer {torch_module_idx} ({type(torch_mlp_modulelist[torch_module_idx])})")
            torch_module_idx += 1
            
        # 3. Handle LayerNorm
        flax_ln_key = f'LayerNorm_{flax_layer_idx}'
        pytorch_layer_at_current_idx_is_layernorm = (
            torch_module_idx < len(torch_mlp_modulelist) and
            isinstance(torch_mlp_modulelist[torch_module_idx], nn.LayerNorm)
        )

        if flax_ln_key in jax_mlp_params:
            print(f"    JAX has {flax_ln_key}. Checking PyTorch module index {torch_module_idx}.")
            if not pytorch_layer_at_current_idx_is_layernorm:
                raise ValueError(
                    f"PyTorch MLP structure mismatch at {base_key_prefix}{torch_module_idx}. "
                    f"JAX has {flax_ln_key}, but PyTorch does not have LayerNorm at this position. "
                    f"Got {type(torch_mlp_modulelist[torch_module_idx]) if torch_module_idx < len(torch_mlp_modulelist) else 'OutOfBound'}."
                )
            flax_ln_p = jax_mlp_params[flax_ln_key]
            torch_ln_p = convert_flax_layernorm_to_torch_layernorm(flax_ln_p)
            for name, tensor in torch_ln_p.items():
                torch_state_dict_for_mlp[f"{base_key_prefix}{torch_module_idx}.{name}"] = tensor
            print(f"      Loaded LayerNorm for {flax_ln_key} into PyTorch layer {torch_module_idx}")
            torch_module_idx += 1
        elif pytorch_layer_at_current_idx_is_layernorm:
            # JAX does NOT have LayerNorm_X, but PyTorch *does* have a LayerNorm here.
            # This means PyTorch's MLP_torch(layer_norm=True) added it.
            # We should skip this PyTorch LayerNorm as it has no JAX params to load.
            print(f"    INFO: JAX params missing '{flax_ln_key}'. "
                  f"PyTorch MLP has a LayerNorm at index {torch_module_idx} ({base_key_prefix}{torch_module_idx}) which will be skipped (uses its PyTorch init).")
            torch_module_idx += 1
        else:
            print(f"    JAX params missing '{flax_ln_key}' and PyTorch has no LayerNorm at index {torch_module_idx}. Continuing.")

        flax_layer_idx += 1
    
    print(f"  [MLP Convert END] Processed {flax_layer_idx} JAX Dense layers. Final PyTorch module index: {torch_module_idx}")
    return torch_state_dict_for_mlp

def unfreeze_and_npify(pytree: Any) -> Any:
    def map_leaf(leaf):
        if isinstance(leaf, (jax.Array, jax.numpy.ndarray)): # Handle both JAX array types
            return np.asarray(jax.device_get(leaf))
        return leaf
    np_pytree = jax.tree_util.tree_map(map_leaf, pytree)
    try: 
        # Attempt to fully unfreeze. For newer Flax, from_state_dict might work well.
        unfrozen_pytree = flax.serialization.from_state_dict(np_pytree, np_pytree) 
        # Ensure all nested FrozenDicts are converted
        unfrozen_pytree = jax.tree_util.tree_map(
            lambda x: dict(x) if isinstance(x, flax.core.FrozenDict) else x,
            unfrozen_pytree,
            is_leaf=lambda x: isinstance(x, flax.core.FrozenDict)
        )
    except Exception: # Fallback if from_state_dict causes issues or for older structures
        unfrozen_pytree = jax.tree_util.tree_map(
            lambda x: dict(x) if isinstance(x, flax.core.FrozenDict) else x,
            np_pytree,
            is_leaf=lambda x: isinstance(x, flax.core.FrozenDict)
        )
    return unfrozen_pytree


def load_gciql_jax_checkpoint_to_pytorch(
    jax_checkpoint_path: str,
    pytorch_agent: Any, # Should be GCIQLAgent_torch instance
):
    print(f"\nLoading GCIQL JAX checkpoint (CONTINUOUS ONLY) from: {jax_checkpoint_path}")
    with open(jax_checkpoint_path, 'rb') as f:
        raw_loaded_dict = pickle.load(f)

    if not ('agent' in raw_loaded_dict and \
            isinstance(raw_loaded_dict['agent'], (dict, flax.core.FrozenDict)) and \
            'network' in raw_loaded_dict['agent'] and \
            isinstance(raw_loaded_dict['agent']['network'], (dict, flax.core.FrozenDict)) and \
            'params' in raw_loaded_dict['agent']['network']):
        print("Checkpoint structure is unexpected. Full loaded dict keys:", list(raw_loaded_dict.keys()))
        if 'agent' in raw_loaded_dict and isinstance(raw_loaded_dict['agent'], (dict, flax.core.FrozenDict)):
            print("'agent' keys:", list(raw_loaded_dict['agent'].keys()))
            if 'network' in raw_loaded_dict['agent'] and isinstance(raw_loaded_dict['agent']['network'], (dict, flax.core.FrozenDict)):
                print("'agent.network' keys:", list(raw_loaded_dict['agent']['network'].keys()))
        raise ValueError("JAX checkpoint structure does not match expected 'agent.network.params' path.")
    
    raw_jax_params = raw_loaded_dict['agent']['network']['params']
    jax_params = unfreeze_and_npify(raw_jax_params)
    print("Successfully extracted and processed JAX parameters from raw checkpoint.")
    print(f"Available top-level module keys in JAX parameters: {list(jax_params.keys())}") # DIAGNOSTIC PRINT

    final_pytorch_state_dict = {}
    
    # --- 1. Value Network (V-function) ---
    print("\n--- Converting Value Network (value) ---")
    expected_value_key = 'modules_value'
    jax_value_module_p = jax_params.get(expected_value_key)
    if jax_value_module_p is None: 
        print(f"ERROR: Missing '{expected_value_key}' in JAX params. Available keys: {list(jax_params.keys())}")
        raise KeyError(f"Missing '{expected_value_key}' in JAX params")

    jax_actual_value_mlp_params = jax_value_module_p.get('value_net')
    if jax_actual_value_mlp_params is None:
        print(f"ERROR: Missing 'value_net' within '{expected_value_key}'. Available keys in '{expected_value_key}': {list(jax_value_module_p.keys())}")
        raise KeyError(f"Missing 'value_net' within '{expected_value_key}'")

    torch_value_mlp_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_actual_value_mlp_params, pytorch_agent.value.mlp.layers, "value.mlp.layers." # Pass the correct dict
    )
    final_pytorch_state_dict.update(torch_value_mlp_sd)
    print(f"  Converted value network from JAX key '{expected_value_key}/value_net'.")


    # --- 2. Critic Network (Q-function, Continuous) ---
    print("\n--- Converting Critic Network (critic) ---")
    expected_critic_key = 'modules_critic'
    jax_critic_module_p = jax_params.get(expected_critic_key) # This is {'value_net': {ensemble params ...}}
    if jax_critic_module_p is None:
        print(f"ERROR: Missing '{expected_critic_key}' in JAX params. Available keys: {list(jax_params.keys())}")
        raise KeyError(f"Missing '{expected_critic_key}' in JAX params")

    jax_critic_ensemble_params_dict = jax_critic_module_p.get('value_net')
    if jax_critic_ensemble_params_dict is None:
        print(f"ERROR: Missing 'value_net' within '{expected_critic_key}'. Available keys in '{expected_critic_key}': {list(jax_critic_module_p.keys())}")
        raise KeyError(f"Missing 'value_net' within '{expected_critic_key}'")

    # jax_critic_ensemble_params_dict now holds the dict with ensemble weights, e.g., {'Dense_0': {'kernel': [2,...]}}
    jax_critic_mlp1_params = jax.tree_util.tree_map(lambda x: x[0], jax_critic_ensemble_params_dict)
    torch_critic_mlp1_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_critic_mlp1_params, pytorch_agent.critic.mlp1.layers, "critic.mlp1.layers."
    )
    final_pytorch_state_dict.update(torch_critic_mlp1_sd)

    jax_critic_mlp2_params = jax.tree_util.tree_map(lambda x: x[1], jax_critic_ensemble_params_dict)
    torch_critic_mlp2_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_critic_mlp2_params, pytorch_agent.critic.mlp2.layers, "critic.mlp2.layers."
    )
    final_pytorch_state_dict.update(torch_critic_mlp2_sd)
    print(f"  Converted critic network (mlp1 and mlp2) from JAX key '{expected_critic_key}/value_net'.")


    # --- 3. Target Critic Network ---
    print("\n--- Converting Target Critic Network (target_critic) ---")
    expected_target_critic_key = 'modules_target_critic'
    jax_target_critic_module_p = jax_params.get(expected_target_critic_key)
    if jax_target_critic_module_p is None:
        print(f"ERROR: Missing '{expected_target_critic_key}' in JAX params. Available keys: {list(jax_params.keys())}")
        raise KeyError(f"Missing '{expected_target_critic_key}' in JAX params")

    jax_target_critic_ensemble_params_dict = jax_target_critic_module_p.get('value_net')
    if jax_target_critic_ensemble_params_dict is None:
        print(f"ERROR: Missing 'value_net' within '{expected_target_critic_key}'. Available keys in '{expected_target_critic_key}': {list(jax_target_critic_module_p.keys())}")
        raise KeyError(f"Missing 'value_net' within '{expected_target_critic_key}'")

    jax_target_critic_mlp1_params = jax.tree_util.tree_map(lambda x: x[0], jax_target_critic_ensemble_params_dict)
    torch_target_critic_mlp1_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_target_critic_mlp1_params, pytorch_agent.target_critic.mlp1.layers, "target_critic.mlp1.layers."
    )
    final_pytorch_state_dict.update(torch_target_critic_mlp1_sd)

    jax_target_critic_mlp2_params = jax.tree_util.tree_map(lambda x: x[1], jax_target_critic_ensemble_params_dict)
    torch_target_critic_mlp2_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_target_critic_mlp2_params, pytorch_agent.target_critic.mlp2.layers, "target_critic.mlp2.layers."
    )
    final_pytorch_state_dict.update(torch_target_critic_mlp2_sd)
    print(f"  Converted target_critic network (mlp1 and mlp2) from JAX key '{expected_target_critic_key}/value_net'.")

    # --- 4. Actor Network (Continuous) ---
    print("\n--- Converting Actor Network (actor) ---")
    expected_actor_key = 'modules_actor'
    jax_actor_module_p = jax_params.get(expected_actor_key)
    if jax_actor_module_p is None: 
        print(f"ERROR: Missing '{expected_actor_key}' in JAX params. Available keys: {list(jax_params.keys())}")
        raise KeyError(f"Missing '{expected_actor_key}' in JAX params")

    ACTOR_MLP_KEY = 'actor_net' 
    MEAN_NET_KEY = 'mean_net'
    LOG_STD_PARAM_KEY = 'log_std_param'

    if ACTOR_MLP_KEY not in jax_actor_module_p:
        print(f"ERROR: JAX actor params ('{expected_actor_key}') missing '{ACTOR_MLP_KEY}'. Available sub-keys: {list(jax_actor_module_p.keys())}")
        raise KeyError(f"JAX actor params ('{expected_actor_key}') missing '{ACTOR_MLP_KEY}'")
    jax_actor_net_params = jax_actor_module_p[ACTOR_MLP_KEY]
    
    torch_actor_net_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_actor_net_params, pytorch_agent.actor.actor_net.layers, "actor.actor_net.layers."
    )
    final_pytorch_state_dict.update(torch_actor_net_sd)

    if MEAN_NET_KEY not in jax_actor_module_p:
        print(f"ERROR: JAX actor params ('{expected_actor_key}') missing '{MEAN_NET_KEY}'. Available sub-keys: {list(jax_actor_module_p.keys())}")
        raise KeyError(f"JAX actor params ('{expected_actor_key}') missing '{MEAN_NET_KEY}'")
    jax_mean_net_params = jax_actor_module_p[MEAN_NET_KEY]
    
    torch_mean_net_p = convert_flax_dense_to_torch_linear(jax_mean_net_params)
    for name, tensor in torch_mean_net_p.items():
        final_pytorch_state_dict[f"actor.mean_net.{name}"] = tensor
    
    if hasattr(pytorch_agent.actor, 'use_learnable_log_std_param') and \
       pytorch_agent.actor.use_learnable_log_std_param:
        if LOG_STD_PARAM_KEY in jax_actor_module_p:
            log_std_val = torch.from_numpy(np.array(jax_actor_module_p[LOG_STD_PARAM_KEY])).float()
            final_pytorch_state_dict["actor.log_stds_param"] = log_std_val
            print(f"  Loaded actor.log_stds_param from JAX key '{expected_actor_key}/{LOG_STD_PARAM_KEY}'.")
        else:
            print(f"  INFO: PyTorch actor expects learnable 'log_stds_param', but not found in JAX params under '{expected_actor_key}/{LOG_STD_PARAM_KEY}'. Using PyTorch default init (zeros).")
    print(f"  Converted actor network from JAX key '{expected_actor_key}'.")

    print(f"\n--- Loading {len(final_pytorch_state_dict)} Converted Parameters into PyTorch GCIQL Agent ---")
    try:
        missing_keys, unexpected_keys = pytorch_agent.load_state_dict(final_pytorch_state_dict, strict=True)
        if missing_keys: print(f"Warning: Missing keys in PyTorch model: {sorted(list(missing_keys))}")
        if unexpected_keys: print(f"Warning: Unexpected keys from JAX checkpoint: {sorted(list(unexpected_keys))}")
        if not missing_keys and not unexpected_keys: print("Successfully loaded all parameters into PyTorch model!")
    except RuntimeError as e:
        print(f"RuntimeError during load_state_dict: {e}")
        torch_model_keys = set(pytorch_agent.state_dict().keys())
        converted_params_keys = set(final_pytorch_state_dict.keys())
        
        print("\n--- PyTorch Model State Dict Keys (for debugging) ---")
        for k in sorted(list(torch_model_keys)): print(k)
        print("\n--- Converted JAX Params Keys (for debugging) ---")
        for k in sorted(list(converted_params_keys)): print(k)

        print("\nKeys expected by PyTorch model but MISSING in converted JAX params:")
        for k_m in sorted(list(torch_model_keys - converted_params_keys)): print(k_m)
        print("\nKeys in converted JAX params but UNEXPECTED by PyTorch model:")
        for k_u in sorted(list(converted_params_keys - torch_model_keys)): print(k_u)
        raise e
    print("--- GCIQL Conversion and Loading Complete (Continuous Only) ---")