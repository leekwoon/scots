import numpy as np
from tqdm import trange
from scipy.spatial.distance import cdist

from cleandiffuser_ex.hilp import HILP
from cleandiffuser_ex.faiss_index_wrapper import FaissIndexWrapper


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    mean_s = np.mean(scores)
    std_s = np.std(scores)
    if std_s > 1e-8:
        return (scores - mean_s) / std_s
    else:
        return np.zeros_like(scores)


def stitch_single_rollout(
    initial_obs: np.ndarray,
    z_direction: np.ndarray,     
    use_hilp: bool,             
    num_steps: int,
    hilp_model: HILP,           
    faiss_index: FaissIndexWrapper, 
    full_traj_dataset: np.ndarray, 
    k_neighbors: int,           
    k_density: int,             
    alpha: float,               
    beta: float,                
    pbar_desc: str = "Rollout Step"
):
    """
    Executes a rollout using a fixed direction vector (z_direction) and a novelty score.
    If use_hilp=True: Progress(latent), Novelty(latent) are calculated.
    If use_hilp=False: Progress(obs), Novelty(obs) are calculated.
    Faiss search is always performed in the latent space.
    Duplicate first frames are removed when connecting segments.

    Args:
        initial_obs (np.ndarray): Starting observation (1, obs_dim).
        z_direction (np.ndarray): Direction vector to use (dimension must match based on use_hilp).
        use_hilp (bool): If True, scores are calculated in latent space; if False, in observation space.
        num_steps (int): Number of steps (segments) to generate.
        hilp_model (HILP): HILP model object (must have a get_phi method).
        faiss_index (FaissIndexWrapper): Pre-calculated Faiss index object (assumed to be based on latent space).
        full_traj_dataset (np.ndarray): Filtered full trajectory dataset (N, horizon, obs_dim).
        k_neighbors (int): Value of k to use when searching for candidate segments.
        k_density (int): Value of k for k-NN when calculating novelty.
        alpha (float): Weight for the Progress score.
        beta (float): Weight for the Novelty score.
        pbar_desc (str): Description string for the progress bar.

    """
    if not isinstance(hilp_model, HILP) or not hasattr(hilp_model, 'get_phi'):
        print("Error: Invalid hilp_model passed.")
        return None
    if not isinstance(faiss_index, FaissIndexWrapper) or not hasattr(faiss_index, 'search'):
         print("Error: Invalid faiss_index passed.")
         return None

    current_obs = initial_obs.copy()
    chosen_segments = []
    visited_history = [] 

    latent_dim = hilp_model.skill_dim
    obs_dim = initial_obs.shape[1]
    if use_hilp and z_direction.shape[0] != latent_dim:
        raise ValueError(f"use_hilp=True requires z_direction dim {latent_dim}, got {z_direction.shape[0]}")
    if not use_hilp and z_direction.shape[0] != obs_dim:
        raise ValueError(f"use_hilp=False requires z_direction dim {obs_dim}, got {z_direction.shape[0]}")

    if use_hilp:
        visited_history.append(hilp_model.get_phi(current_obs)[0]) 
    else:
        visited_history.append(current_obs[0]) 

    step_iterator = trange(num_steps, desc=pbar_desc, leave=False) if 'trange' in locals() else range(num_steps)

    for step in step_iterator:
        current_latent = hilp_model.get_phi(current_obs)
        distances, idxs = faiss_index.search(current_latent, k=k_neighbors)

        neighbor_indices = idxs[0]
        valid_mask = neighbor_indices != -1
        valid_neighbor_indices = neighbor_indices[valid_mask]

        neighbors_original_trajs = full_traj_dataset[valid_neighbor_indices]
        k_valid = neighbors_original_trajs.shape[0]

        progress_scores = np.zeros(k_valid, dtype=np.float32)
        novelty_scores = np.zeros(k_valid, dtype=np.float32)
        total_scores = np.full(k_valid, -np.inf, dtype=np.float32)
        phi_ends_batch = None 

        start_obs_batch = neighbors_original_trajs[:, 0, :]
        end_obs_batch = neighbors_original_trajs[:, -1, :]

        if use_hilp:
            phi_starts_batch = hilp_model.get_phi(start_obs_batch)
            phi_ends_batch = hilp_model.get_phi(end_obs_batch)
            delta_phi = phi_ends_batch - phi_starts_batch
            progress_scores = np.dot(delta_phi, z_direction) 
        else:
            delta_obs = end_obs_batch - start_obs_batch
            progress_scores = np.dot(delta_obs, z_direction) 

        num_visited = len(visited_history)
        visited_array = np.array(visited_history) # (N_visited, dim)
        current_k_density = min(k_density, num_visited)

        if use_hilp:
            # Novelty in LATENT space
            if phi_ends_batch is None: 
                phi_ends_batch = hilp_model.get_phi(end_obs_batch)
            if visited_array.shape[1] != latent_dim: raise ValueError("Visited history should contain latents when use_hilp=True")
            dist_matrix = cdist(phi_ends_batch, visited_array, metric='euclidean')
        else:
            # Novelty in OBSERVATION space
            if visited_array.shape[1] != obs_dim: raise ValueError("Visited history should contain observations when use_hilp=False")
            dist_matrix = cdist(end_obs_batch, visited_array, metric='euclidean')

        knn_indices = np.argsort(dist_matrix, axis=1)[:, :current_k_density]
        knn_distances = np.take_along_axis(dist_matrix, knn_indices, axis=1)
        novelty_scores = np.mean(knn_distances, axis=1)

        p_norm = normalize_scores(progress_scores)
        n_norm = normalize_scores(novelty_scores) 
        total_scores = alpha * p_norm + beta * n_norm

        valid_score_mask = np.isfinite(total_scores)
        if not np.any(valid_score_mask):
            best_idx = 0 
            print(f" Warning: Step {step+1}: All scores invalid. Selecting first neighbor.")
        else:
            valid_scores = total_scores[valid_score_mask]
            best_idx_in_valid = np.argmax(valid_scores)
            best_idx = np.where(valid_score_mask)[0][best_idx_in_valid]

        best_segment = neighbors_original_trajs[best_idx]

        if step == 0:
            chosen_segments.append(best_segment)
        else:
            chosen_segments.append(best_segment[1:])

        current_obs = best_segment[-1:, :] 

        if use_hilp:
            if phi_ends_batch is not None and best_idx < phi_ends_batch.shape[0]:
                visited_history.append(phi_ends_batch[best_idx])
            else: 
                visited_history.append(hilp_model.get_phi(current_obs)[0])
        else:
            visited_history.append(current_obs[0]) 

        if hasattr(step_iterator, 'set_description'):
            step_iterator.set_description(f"{pbar_desc} -> {current_obs[0,:2].round(2)}")

    return np.concatenate(chosen_segments, axis=0)


