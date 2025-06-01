import pickle
import numpy as np
from typing import Dict, Mapping, Any

import torch
import torch.nn as nn

import jax
import jax.numpy as jnp # Often needed by pickle for JAX arrays if not pre-converted
import flax.serialization # Needed to load the JAX state dict properly


def convert_flax_dense_to_torch_linear(flax_params: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Converts Flax Dense layer params (kernel, bias) to PyTorch Linear (weight, bias)."""
    torch_params = {}
    if 'kernel' in flax_params:
        kernel_np = np.array(flax_params['kernel'])
        torch_params['weight'] = torch.from_numpy(np.transpose(kernel_np)).float()
    if 'bias' in flax_params:
        torch_params['bias'] = torch.from_numpy(np.array(flax_params['bias'])).float()
    return torch_params

def convert_flax_layernorm_to_torch_layernorm(flax_params: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Converts Flax LayerNorm params (scale, bias) to PyTorch LayerNorm (weight, bias)."""
    torch_params = {}
    if 'scale' in flax_params: # Flax uses 'scale', PyTorch uses 'weight'
        torch_params['weight'] = torch.from_numpy(np.array(flax_params['scale'])).float()
    if 'bias' in flax_params:
        torch_params['bias'] = torch.from_numpy(np.array(flax_params['bias'])).float()
    return torch_params

def convert_jax_mlp_to_torch_mlp(jax_mlp_params: Dict, torch_mlp_layers: nn.ModuleList):
    """
    Converts parameters for a Flax MLP to a PyTorch MLP (nn.ModuleList).
    ASSUMES JAX parameter keys are named 'Dense_0', 'LayerNorm_0', 'Dense_1', etc.
    """
    print(f"  [convert_jax_mlp_to_torch_mlp] Attempting conversion assuming 'Dense_i' JAX keys.")
    print(f"  Input JAX keys: {list(jax_mlp_params.keys())}") # Debug: Print received keys

    torch_state_dict = {}
    torch_layer_idx = 0 # Index for iterating through PyTorch nn.ModuleList layers
    flax_layer_idx = 0  # Index for iterating through JAX 'Dense_i' layers

    # Iterate while the expected JAX Dense layer key exists
    while f'Dense_{flax_layer_idx}' in jax_mlp_params:
        print(f"    Processing Flax 'Dense_{flax_layer_idx}'...")
        flax_dense_params = jax_mlp_params[f'Dense_{flax_layer_idx}']

        # Find corresponding nn.Linear in torch_mlp_layers
        target_torch_layer_type = nn.Linear
        target_torch_layer = None
        start_search_idx = torch_layer_idx
        while start_search_idx < len(torch_mlp_layers):
            if isinstance(torch_mlp_layers[start_search_idx], target_torch_layer_type):
                target_torch_layer = torch_mlp_layers[start_search_idx]
                torch_layer_idx = start_search_idx # Update main index
                break
            start_search_idx += 1
        if target_torch_layer is None:
             raise ValueError(f"Could not find corresponding PyTorch {target_torch_layer_type.__name__} for Flax Dense_{flax_layer_idx}")

        print(f"      Mapping JAX Dense_{flax_layer_idx} to PyTorch Linear layer {torch_layer_idx}")
        torch_prefix = f'layers.{torch_layer_idx}.'
        converted_params = convert_flax_dense_to_torch_linear(flax_dense_params)
        for k, v in converted_params.items():
            torch_state_dict[torch_prefix + k] = v
        torch_layer_idx += 1 # Advance torch index past the matched Linear layer

        # --- Check for Optional LayerNorm ---
        # Assumes LayerNorm (if present) immediately follows Dense in JAX params
        flax_ln_key = f'LayerNorm_{flax_layer_idx}'
        if flax_ln_key in jax_mlp_params:
            print(f"    Processing Flax 'LayerNorm_{flax_layer_idx}'...")
            flax_ln_params = jax_mlp_params[flax_ln_key]

            # Find corresponding nn.LayerNorm in torch_mlp_layers
            target_torch_layer_type = nn.LayerNorm
            target_torch_layer = None
            start_search_idx = torch_layer_idx
            while start_search_idx < len(torch_mlp_layers):
                if isinstance(torch_mlp_layers[start_search_idx], target_torch_layer_type):
                    target_torch_layer = torch_mlp_layers[start_search_idx]
                    torch_layer_idx = start_search_idx # Update main index
                    break
                start_search_idx += 1
            if target_torch_layer is None:
                 raise ValueError(f"Could not find corresponding PyTorch {target_torch_layer_type.__name__} for Flax LayerNorm_{flax_layer_idx}")

            print(f"      Mapping JAX LayerNorm_{flax_layer_idx} to PyTorch LayerNorm layer {torch_layer_idx}")
            torch_prefix = f'layers.{torch_layer_idx}.'
            converted_ln_params = convert_flax_layernorm_to_torch_layernorm(flax_ln_params)
            for k, v in converted_ln_params.items():
                torch_state_dict[torch_prefix + k] = v
            torch_layer_idx += 1 # Advance torch index past the matched LayerNorm layer

        # --- Advance past PyTorch Activation Layer ---
        # Assumes activation (like GELU) follows Linear/LayerNorm in PyTorch ModuleList
        if torch_layer_idx < len(torch_mlp_layers) and not isinstance(torch_mlp_layers[torch_layer_idx], (nn.Linear, nn.LayerNorm)):
             print(f"      Advancing PyTorch index {torch_layer_idx} past non-Linear/LayerNorm layer (Activation?).")
             torch_layer_idx += 1

        flax_layer_idx += 1 # Move to the next expected Flax Dense layer index

    print(f"  [convert_jax_mlp_to_torch_mlp] Finished conversion. Returning dict with {len(torch_state_dict)} parameter tensors.")
    return torch_state_dict

def load_hilp_jax_checkpoint_to_pytorch(
    jax_checkpoint_path: str,
    pytorch_agent: nn.Module, # Expecting HILP_torch which is an nn.Module
):
    """Loads parameters from a JAX HILP checkpoint into a PyTorch HILP agent."""

    print(f"\nLoading HILP JAX checkpoint from: {jax_checkpoint_path}")
    with open(jax_checkpoint_path, 'rb') as f:
        jax_loaded_dict = pickle.load(f)

    # --- Extract JAX Parameters ---
    if 'agent' in jax_loaded_dict and isinstance(jax_loaded_dict['agent'], Mapping):
        try:
            jax_params = jax_loaded_dict['agent']['network']['params']
            print("Extracted jax_params via direct dictionary access.")
        except (KeyError, TypeError):
             print("Direct access failed, trying flax.serialization.from_state_dict...")
             try:
                 target_state_example = {'network': {'params': {}}}
                 restored_agent_state = flax.serialization.from_state_dict(target_state_example, jax_loaded_dict['agent'])
                 jax_params = restored_agent_state['network']['params']
                 print("Extracted jax_params using from_state_dict.")
             except Exception as e:
                  raise ValueError(f"Could not extract JAX parameters from checkpoint: {e}")
    else:
        raise ValueError("Unexpected JAX checkpoint structure. Expected {'agent': ...}")

    print("\nJAX parameters loaded/extracted, starting conversion...")
    pytorch_state_dict = {} # This will store the final converted parameters

    print("\n--- Converting Value Network ---")
    try:
        jax_value_phi_ensemble_params = jax_params['modules_value']['phi']['VmapMLP_0']
        print("Accessing JAX params under ['modules_value']['phi']['VmapMLP_0']")
    except KeyError as e:
         print(f"\nFatal Error: Could not access 'VmapMLP_0' key under ['modules_value']['phi']. Check checkpoint structure.")
         if 'modules_value' in jax_params and 'phi' in jax_params['modules_value']:
             print(f"Keys available under ['modules_value']['phi']: {list(jax_params['modules_value']['phi'].keys())}")
         raise e

    print("Converting Ensemble Member 1...")
    jax_value1_params = jax.tree_map(lambda x: np.array(x[0]), jax_value_phi_ensemble_params)
    torch_mlp1_params = convert_jax_mlp_to_torch_mlp(jax_value1_params, pytorch_agent.value.phi_net.mlp1.layers)
    for k, v in torch_mlp1_params.items():
        pytorch_state_dict[f'value.phi_net.mlp1.{k}'] = v # Add to final state dict

    print("\nConverting Ensemble Member 2...")
    jax_value2_params = jax.tree_map(lambda x: np.array(x[1]), jax_value_phi_ensemble_params)
    torch_mlp2_params = convert_jax_mlp_to_torch_mlp(jax_value2_params, pytorch_agent.value.phi_net.mlp2.layers)
    for k, v in torch_mlp2_params.items():
        pytorch_state_dict[f'value.phi_net.mlp2.{k}'] = v # Add to final state dict
    print("--- Value Network (phi) Conversion Complete ---")

    print("\n--- Converting Target Value Network ---")
    try:
        jax_target_value_phi_ensemble_params = jax_params['modules_target_value']['phi']['VmapMLP_0']
        print("Accessing JAX params under ['modules_target_value']['phi']['VmapMLP_0']")
    except KeyError as e:
         print(f"\nFatal Error: Could not access 'VmapMLP_0' key under ['modules_target_value']['phi']. Check checkpoint structure.")
         raise e

    print("Converting Target Ensemble Member 1...")
    jax_target_value1_params = jax.tree_map(lambda x: np.array(x[0]), jax_target_value_phi_ensemble_params)
    torch_target_mlp1_params = convert_jax_mlp_to_torch_mlp(jax_target_value1_params, pytorch_agent.target_value.phi_net.mlp1.layers)
    for k, v in torch_target_mlp1_params.items():
        pytorch_state_dict[f'target_value.phi_net.mlp1.{k}'] = v

    print("\nConverting Target Ensemble Member 2...")
    jax_target_value2_params = jax.tree_map(lambda x: np.array(x[1]), jax_target_value_phi_ensemble_params)
    torch_target_mlp2_params = convert_jax_mlp_to_torch_mlp(jax_target_value2_params, pytorch_agent.target_value.phi_net.mlp2.layers)
    for k, v in torch_target_mlp2_params.items():
        pytorch_state_dict[f'target_value.phi_net.mlp2.{k}'] = v
    print("--- Target Value Network (phi) Conversion Complete ---")

    print("\n--- Loading Converted State Dict into PyTorch Agent ---")
    print(f"Total items in final pytorch_state_dict: {len(pytorch_state_dict)}")

    missing_keys, unexpected_keys = pytorch_agent.load_state_dict(pytorch_state_dict, strict=True)
    print(missing_keys, unexpected_keys)
