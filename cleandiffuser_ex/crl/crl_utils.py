import pickle
import flax.core # For FrozenDict type hint
import flax.serialization
import jax # For tree_map and Array types
import jax.numpy # For Array type
import jax.tree_util # For tree_map
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Sequence, Optional, Callable, Tuple


# --- Helper functions (reuse from GCIQL loader) ---
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
    if 'scale' in flax_params: # JAX LayerNorm uses 'scale'
        torch_params['weight'] = torch.from_numpy(np.array(flax_params['scale'])).float()
    if 'bias' in flax_params:
        torch_params['bias'] = torch.from_numpy(np.array(flax_params['bias'])).float()
    return torch_params


def convert_jax_mlp_to_torch_mlp_modulelist(
    jax_mlp_params: Dict[str, Any], # This should be the dict containing 'Dense_0', 'LayerNorm_0' etc.
    torch_mlp_modulelist: nn.ModuleList, # The .layers attribute of an MLP_torch instance
    base_key_prefix: str # e.g., "critic.phi1.layers."
    ) -> Dict[str, torch.Tensor]:
    """Converts parameters of a JAX MLP (like one inside GCBilinearValue) to a PyTorch state_dict."""
    torch_state_dict_for_mlp = {}
    flax_layer_idx = 0
    torch_module_idx = 0
    processed_flax_keys = set()

    print(f"  [MLP Convert START] JAX keys: {list(jax_mlp_params.keys())}, PyTorch num_modules: {len(torch_mlp_modulelist)}, Prefix: {base_key_prefix}")

    # Iterate through PyTorch modules, expecting corresponding JAX params
    while torch_module_idx < len(torch_mlp_modulelist):
        torch_module = torch_mlp_modulelist[torch_module_idx]
        print(f"    Processing PyTorch module index {torch_module_idx}: {type(torch_module)}")

        if isinstance(torch_module, nn.Linear):
            # Expecting a corresponding Dense layer in JAX
            flax_dense_key = f'Dense_{flax_layer_idx}'
            if flax_dense_key not in jax_mlp_params:
                 # Check if JAX MLP ended early (e.g. output_dim=None in PyTorch led to fewer layers than expected JAX structure)
                 # Or maybe PyTorch MLP has extra layers?
                 print(f"    WARNING: PyTorch has Linear at index {torch_module_idx}, but JAX params missing '{flax_dense_key}'. Stopping MLP conversion here for {base_key_prefix}.")
                 # Should we raise an error or just stop? Stopping might be safer.
                 break # Exit the loop for this MLP
                 # raise ValueError(f"PyTorch MLP structure mismatch at {base_key_prefix}{torch_module_idx}. Expected JAX params for '{flax_dense_key}'.")

            print(f"      Matching JAX '{flax_dense_key}'...")
            flax_dense_p = jax_mlp_params[flax_dense_key]
            torch_linear_p = convert_flax_dense_to_torch_linear(flax_dense_p)
            for name, tensor in torch_linear_p.items():
                torch_state_dict_for_mlp[f"{base_key_prefix}{torch_module_idx}.{name}"] = tensor
            print(f"      Loaded Linear from '{flax_dense_key}' into PyTorch layer {torch_module_idx}")
            processed_flax_keys.add(flax_dense_key)
            flax_layer_idx += 1 # Increment JAX layer index *only after* processing a Dense layer
            torch_module_idx += 1

        elif isinstance(torch_module, nn.LayerNorm):
            # Expecting a corresponding LayerNorm *after* the previous Dense in JAX
            # JAX LayerNorm index matches the *preceding* Dense index
            flax_ln_key = f'LayerNorm_{flax_layer_idx - 1}' # Use previous flax_layer_idx
            if flax_ln_key in jax_mlp_params:
                print(f"      Matching JAX '{flax_ln_key}'...")
                flax_ln_p = jax_mlp_params[flax_ln_key]
                torch_ln_p = convert_flax_layernorm_to_torch_layernorm(flax_ln_p)
                for name, tensor in torch_ln_p.items():
                    torch_state_dict_for_mlp[f"{base_key_prefix}{torch_module_idx}.{name}"] = tensor
                print(f"      Loaded LayerNorm from '{flax_ln_key}' into PyTorch layer {torch_module_idx}")
                processed_flax_keys.add(flax_ln_key)
                torch_module_idx += 1
            else:
                # PyTorch has LayerNorm, but JAX doesn't for this position.
                # This can happen if PyTorch MLP was created with layer_norm=True
                # but the specific JAX module didn't use it at that point.
                print(f"      INFO: PyTorch has LayerNorm at index {torch_module_idx}, but JAX params missing '{flax_ln_key}'. Skipping PyTorch layer (uses its PyTorch init).")
                torch_module_idx += 1

        elif isinstance(torch_module, (nn.GELU, nn.ReLU, nn.Tanh)): # Add other activations if used
            # Skip activation layers, no parameters to load
            print(f"      Skipping PyTorch activation layer {torch_module_idx} ({type(torch_module)})")
            torch_module_idx += 1

        else:
            # Unexpected module type in PyTorch MLP
            raise TypeError(f"Unexpected module type {type(torch_module)} found in PyTorch MLP at index {torch_module_idx} for prefix {base_key_prefix}")

    # Check if all JAX keys were processed
    remaining_jax_keys = set(jax_mlp_params.keys()) - processed_flax_keys
    if remaining_jax_keys:
        print(f"  [MLP Convert WARNING] The following JAX keys were not used for {base_key_prefix}: {sorted(list(remaining_jax_keys))}")

    print(f"  [MLP Convert END] Processed up to PyTorch module index {torch_module_idx-1}. Final JAX layer index: {flax_layer_idx-1}")
    return torch_state_dict_for_mlp


def unfreeze_and_npify(pytree: Any) -> Any:
    """Converts JAX/Flax FrozenDicts/Arrays to nested Python dicts/NumPy arrays."""
    def map_leaf(leaf):
        if isinstance(leaf, (jax.Array, jax.numpy.ndarray)):
            return np.asarray(jax.device_get(leaf))
        return leaf

    def unfreeze_recursive(node):
         if isinstance(node, flax.core.FrozenDict):
              # Convert FrozenDict to dict, recursively processing values
              return {k: unfreeze_recursive(v) for k, v in node.items()}
         elif isinstance(node, (list, tuple)):
              # Recursively process elements in lists/tuples
              return type(node)(unfreeze_recursive(item) for item in node)
         elif isinstance(node, dict):
             # Recursively process values in regular dicts
             return {k: unfreeze_recursive(v) for k, v in node.items()}
         else:
              # Apply npify to leaves (like JAX arrays)
              return map_leaf(node)

    return unfreeze_recursive(pytree)


# --- Main Loading Function ---
def load_crl_jax_checkpoint_to_pytorch(
    jax_checkpoint_path,
    pytorch_agent, # Must be an initialized CRLAgent_torch instance
):
    """Loads weights from a saved JAX CRLAgent checkpoint into a PyTorch CRLAgent_torch."""
    print(f"\nLoading CRL JAX checkpoint (CONTINUOUS ONLY) from: {jax_checkpoint_path}")
    with open(jax_checkpoint_path, 'rb') as f:
        # Use encoding='latin1' if you encounter UnicodeDecodeError with Python 3+ loading Python 2 pickles
        try:
            raw_loaded_dict = pickle.load(f)
        except UnicodeDecodeError:
            print("UnicodeDecodeError encountered, trying latin1 encoding...")
            f.seek(0) # Reset file pointer
            raw_loaded_dict = pickle.load(f, encoding='latin1')


    # --- Parameter Extraction and Conversion ---
    if not ('agent' in raw_loaded_dict and
            isinstance(raw_loaded_dict['agent'], (dict, flax.core.FrozenDict)) and
            'network' in raw_loaded_dict['agent'] and
            isinstance(raw_loaded_dict['agent']['network'], (dict, flax.core.FrozenDict)) and
            'params' in raw_loaded_dict['agent']['network']):
        # Try alternative structure seen in some checkpoints: top-level 'agent' is the TrainState
        if isinstance(raw_loaded_dict.get('agent'), (dict, flax.core.FrozenDict)) and \
           'params' in raw_loaded_dict['agent']:
             raw_jax_params = raw_loaded_dict['agent']['params']
             print("Accessed params via top-level 'agent.params'.")
        else:
             print("Checkpoint structure is unexpected. Full loaded dict keys:", list(raw_loaded_dict.keys()))
             if 'agent' in raw_loaded_dict and isinstance(raw_loaded_dict['agent'], (dict, flax.core.FrozenDict)):
                 print("'agent' keys:", list(raw_loaded_dict['agent'].keys()))
                 if 'network' in raw_loaded_dict['agent'] and isinstance(raw_loaded_dict['agent']['network'], (dict, flax.core.FrozenDict)):
                     print("'agent.network' keys:", list(raw_loaded_dict['agent']['network'].keys()))
             raise ValueError("JAX checkpoint structure does not match expected 'agent.network.params' or 'agent.params' path.")
    else: # Standard structure
        raw_jax_params = raw_loaded_dict['agent']['network']['params']
        print("Accessed params via 'agent.network.params'.")

    # Convert JAX arrays to NumPy and unfreeze FrozenDicts
    jax_params = unfreeze_and_npify(raw_jax_params)
    print("Successfully extracted and processed JAX parameters.")
    print(f"Available top-level module keys in JAX parameters: {list(jax_params.keys())}")

    # --- State Dictionary Construction ---
    final_pytorch_state_dict = {}

    # --- 1. Critic Network (GCBilinearValue, Ensemble) ---
    print("\n--- Converting Critic Network (critic) ---")
    expected_critic_key = 'modules_critic'
    if expected_critic_key not in jax_params:
        raise KeyError(f"Missing '{expected_critic_key}' in JAX params. Available keys: {list(jax_params.keys())}")
    jax_critic_module_p = jax_params[expected_critic_key] # Contains 'phi' and 'psi'

    # Check for 'phi' and 'psi' inside the critic module params
    if 'phi' not in jax_critic_module_p or 'psi' not in jax_critic_module_p:
         raise KeyError(f"Missing 'phi' or 'psi' within '{expected_critic_key}'. Found keys: {list(jax_critic_module_p.keys())}")

    jax_critic_phi_ensemble_params = jax_critic_module_p['phi'] # This has ensemble structure (e.g., Dense_0/kernel shape [2,...])
    jax_critic_psi_ensemble_params = jax_critic_module_p['psi'] # This has ensemble structure

    # Extract ensemble members (0 and 1) for phi and psi MLPs
    # tree_map applies function to leaves (the actual parameter arrays)
    jax_critic_phi0_params = jax.tree_util.tree_map(lambda x: x[0], jax_critic_phi_ensemble_params)
    jax_critic_phi1_params = jax.tree_util.tree_map(lambda x: x[1], jax_critic_phi_ensemble_params)
    jax_critic_psi0_params = jax.tree_util.tree_map(lambda x: x[0], jax_critic_psi_ensemble_params)
    jax_critic_psi1_params = jax.tree_util.tree_map(lambda x: x[1], jax_critic_psi_ensemble_params)

    # Convert and add to state dict
    # Ensemble 0 -> PyTorch mlp1
    torch_critic_phi1_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_critic_phi0_params, pytorch_agent.critic.phi1.layers, "critic.phi1.layers."
    )
    final_pytorch_state_dict.update(torch_critic_phi1_sd)
    torch_critic_psi1_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_critic_psi0_params, pytorch_agent.critic.psi1.layers, "critic.psi1.layers."
    )
    final_pytorch_state_dict.update(torch_critic_psi1_sd)

    # Ensemble 1 -> PyTorch mlp2
    torch_critic_phi2_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_critic_phi1_params, pytorch_agent.critic.phi2.layers, "critic.phi2.layers."
    )
    final_pytorch_state_dict.update(torch_critic_phi2_sd)
    torch_critic_psi2_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_critic_psi1_params, pytorch_agent.critic.psi2.layers, "critic.psi2.layers."
    )
    final_pytorch_state_dict.update(torch_critic_psi2_sd)
    print(f"  Converted critic network (phi1, psi1, phi2, psi2) from JAX key '{expected_critic_key}'.")


    # --- 2. Value Network (GCBilinearValue, Non-Ensemble, Optional) ---
    print("\n--- Converting Value Network (value, if present) ---")
    expected_value_key = 'modules_value'
    if hasattr(pytorch_agent, 'value') and pytorch_agent.value is not None:
        if expected_value_key not in jax_params:
            # If PyTorch agent expects 'value' but it's missing in JAX, raise error.
            # This happens if JAX was saved with ddpgbc and PyTorch model initialized for awr.
             raise KeyError(f"PyTorch agent requires '{expected_value_key}' (for AWR actor_loss), but it's missing in JAX params. Available keys: {list(jax_params.keys())}")

        print(f"  PyTorch agent has 'value' network. Converting from JAX key '{expected_value_key}'...")
        jax_value_module_p = jax_params[expected_value_key] # Contains 'phi' and 'psi'

        if 'phi' not in jax_value_module_p or 'psi' not in jax_value_module_p:
             raise KeyError(f"Missing 'phi' or 'psi' within '{expected_value_key}'. Found keys: {list(jax_value_module_p.keys())}")

        # No ensemble here, directly access the MLP params
        jax_value_phi_params = jax_value_module_p['phi']
        jax_value_psi_params = jax_value_module_p['psi']

        # Convert and add to state dict
        torch_value_phi_sd = convert_jax_mlp_to_torch_mlp_modulelist(
            jax_value_phi_params, pytorch_agent.value.phi.layers, "value.phi.layers."
        )
        final_pytorch_state_dict.update(torch_value_phi_sd)

        torch_value_psi_sd = convert_jax_mlp_to_torch_mlp_modulelist(
            jax_value_psi_params, pytorch_agent.value.psi.layers, "value.psi.layers."
        )
        final_pytorch_state_dict.update(torch_value_psi_sd)
        print(f"  Converted value network (phi, psi) from JAX key '{expected_value_key}'.")

    elif expected_value_key in jax_params:
        # JAX params have 'value', but PyTorch agent doesn't (e.g., JAX saved with awr, PyTorch model initialized for ddpgbc)
        print(f"  INFO: JAX params contain '{expected_value_key}', but PyTorch agent does not have a 'value' network. Skipping conversion.")
    else:
        # Neither JAX nor PyTorch have 'value' network (e.g., both configured for ddpgbc)
        print("  No 'value' network found in JAX params, and PyTorch agent does not expect one. Skipping.")

    # --- 3. Actor Network (GCActor) ---
    print("\n--- Converting Actor Network (actor) ---")
    expected_actor_key = 'modules_actor'
    if expected_actor_key not in jax_params:
        raise KeyError(f"Missing '{expected_actor_key}' in JAX params. Available keys: {list(jax_params.keys())}")
    jax_actor_module_p = jax_params[expected_actor_key] # Contains 'actor_net', 'mean_net', maybe 'log_stds'

    # Define expected keys within the actor module
    ACTOR_MLP_KEY = 'actor_net'
    MEAN_NET_KEY = 'mean_net'
    LOG_STD_PARAM_KEY = 'log_stds' # Note: JAX uses 'log_stds', PyTorch uses 'log_stds_param'

    # Convert Actor MLP ('actor_net')
    if ACTOR_MLP_KEY not in jax_actor_module_p:
        raise KeyError(f"JAX actor params ('{expected_actor_key}') missing '{ACTOR_MLP_KEY}'. Available sub-keys: {list(jax_actor_module_p.keys())}")
    jax_actor_net_params = jax_actor_module_p[ACTOR_MLP_KEY]

    torch_actor_net_sd = convert_jax_mlp_to_torch_mlp_modulelist(
        jax_actor_net_params, pytorch_agent.actor.actor_net.layers, "actor.actor_net.layers."
    )
    final_pytorch_state_dict.update(torch_actor_net_sd)

    # Convert Mean Network ('mean_net') - It's a single Dense layer
    if MEAN_NET_KEY not in jax_actor_module_p:
        raise KeyError(f"JAX actor params ('{expected_actor_key}') missing '{MEAN_NET_KEY}'. Available sub-keys: {list(jax_actor_module_p.keys())}")
    jax_mean_net_params = jax_actor_module_p[MEAN_NET_KEY] # This is the dict for the Dense layer {'kernel': ..., 'bias': ...}

    torch_mean_net_p = convert_flax_dense_to_torch_linear(jax_mean_net_params)
    for name, tensor in torch_mean_net_p.items():
        final_pytorch_state_dict[f"actor.mean_net.{name}"] = tensor

    # Convert Log Standard Deviations ('log_stds'), if applicable
    # This applies only if const_std=False in the config
    if hasattr(pytorch_agent.actor, 'use_learnable_log_std_param') and \
       pytorch_agent.actor.use_learnable_log_std_param:
        if LOG_STD_PARAM_KEY in jax_actor_module_p:
            # Ensure the JAX param is a scalar or 1D array compatible with the PyTorch parameter
            jax_log_std_val = np.array(jax_actor_module_p[LOG_STD_PARAM_KEY])
            # PyTorch param is typically 1D (action_dim,)
            if jax_log_std_val.ndim > 1:
                 raise ValueError(f"JAX '{LOG_STD_PARAM_KEY}' has unexpected shape {jax_log_std_val.shape}. Expected scalar or 1D.")
            # If scalar in JAX, expand it to match PyTorch action_dim
            if jax_log_std_val.ndim == 0:
                 print(f"  INFO: JAX '{LOG_STD_PARAM_KEY}' is a scalar. Expanding to action_dim for PyTorch.")
                 log_std_val = torch.from_numpy(jax_log_std_val).float().expand(pytorch_agent.action_dim)
            else:
                 log_std_val = torch.from_numpy(jax_log_std_val).float()

            # Assign to the correct PyTorch parameter name
            final_pytorch_state_dict["actor.log_stds_param"] = log_std_val
            print(f"  Loaded actor.log_stds_param from JAX key '{expected_actor_key}/{LOG_STD_PARAM_KEY}'.")
        else:
            # PyTorch expects learnable std, but JAX checkpoint (likely const_std=True) doesn't have it.
            # The PyTorch parameter will keep its default init (zeros).
            print(f"  INFO: PyTorch actor expects learnable 'log_stds_param', but not found in JAX params under '{expected_actor_key}/{LOG_STD_PARAM_KEY}'. Using PyTorch default init (zeros).")
    elif LOG_STD_PARAM_KEY in jax_actor_module_p:
         # JAX has log_stds, but PyTorch actor doesn't expect it (const_std=True). Ignore.
         print(f"  INFO: JAX params contain '{expected_actor_key}/{LOG_STD_PARAM_KEY}', but PyTorch actor does not expect learnable stds. Skipping.")

    print(f"  Converted actor network from JAX key '{expected_actor_key}'.")

    # --- Final PyTorch model parameter loading ---
    print(f"\n--- Loading {len(final_pytorch_state_dict)} Converted Parameters into PyTorch CRL Agent ---")
    try:
        # Load the constructed state dictionary into the PyTorch model
        # Set strict=False initially to debug missing/unexpected keys easily
        missing_keys, unexpected_keys = pytorch_agent.load_state_dict(final_pytorch_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in PyTorch model (parameters not found in JAX checkpoint or conversion):")
            for k in sorted(missing_keys): print(f"  - {k}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys found (parameters in converted JAX dict but not in PyTorch model):")
            for k in sorted(unexpected_keys): print(f"  - {k}")

        # If strict loading is desired, uncomment the following check:
        # if missing_keys or unexpected_keys:
        #     raise RuntimeError("State dict mismatch during loading (see warnings above). Set strict=False for potentially partial load.")
        # else:
        #     print("Successfully loaded all parameters strictly into PyTorch model!")

        if not missing_keys and not unexpected_keys:
             print("Successfully loaded all parameters into PyTorch model (strict check passed implicitly)!")
        elif not missing_keys and unexpected_keys:
             print("Successfully loaded parameters into PyTorch model, but some JAX keys were unused.")
        elif missing_keys:
             print("Partially loaded parameters. Some PyTorch parameters retain their initial values.")
        else: # Should not happen if strict=False
             print("Loaded parameters with unexpected status.")

    except RuntimeError as e:
        # Fallback for other runtime errors during loading
        print(f"RuntimeError during load_state_dict: {e}")
        print("This might indicate shape mismatches or other critical errors.")
        # Add more detailed debugging if necessary
        torch_model_keys = set(pytorch_agent.state_dict().keys())
        converted_params_keys = set(final_pytorch_state_dict.keys())
        print("\n--- PyTorch Model State Dict Keys (for debugging) ---")
        for k in sorted(list(torch_model_keys)): print(k)
        print("\n--- Converted JAX Params Keys (for debugging) ---")
        for k in sorted(list(converted_params_keys)): print(k)
        raise e

    print("--- CRL Conversion and Loading Complete (Continuous Only) ---")