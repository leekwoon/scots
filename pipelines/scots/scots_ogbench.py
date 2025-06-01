import os
import csv
import einops
import ogbench
import gymnasium
from ogbench.utils import load_dataset
from collections import defaultdict
import hydra, wandb, uuid, tempfile
import numpy as np
from tqdm import tqdm
from tqdm import trange
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser_ex.diffusion import ContinuousDiffusionSDEEX
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import report_parameters

from cleandiffuser_ex.utils import set_seed
from cleandiffuser_ex.gciql import GCIQL
from cleandiffuser_ex.crl import CRL
from cleandiffuser_ex.dataset.ogbench_dataset import MultiHorizonOGBenchDataset


max_epsode_lengths = {
    "pointmaze-medium-stitch-v0": 1000,
    "pointmaze-large-stitch-v0": 1000,
    "pointmaze-giant-stitch-v0": 2000, 
    "antmaze-medium-stitch-v0": 1000,
    "antmaze-large-stitch-v0": 2000,
    "antmaze-giant-stitch-v0": 2000,
    "antmaze-medium-explore-v0": 2000,
    "antmaze-large-explore-v0": 2000,
}


def make_env_fn(env_id):
    splits = env_id.split('-')
    env_name = '-'.join(splits[:-2] + splits[-1:])
    def _init():
        env = gymnasium.make(env_name)
        env.reset()
        return env
    return _init


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict): # Check if it's a dict
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')



def evaluate(
    diffusions_models,
    low_controller,
    env,
    normalizer,
    task_id, # The identifier the environment's reset() function expects
    planning_horizons,
    obs_dim,
    config, # Your Hydra args object
    num_eval_episodes,
    num_video_episodes,
    video_frame_skip,
):

    
    """
    Evaluates the agent on a single task.
    """
    H_hi = planning_horizons[0]
    H_lo = planning_horizons[1]

    # Templates for planning priors
    high_level_prior_template = torch.zeros((1, H_hi, obs_dim), device=config.device)
    mega_lowlevel_prior_template = torch.zeros(
        (config.num_candidates, (H_hi - 1), H_lo, obs_dim), device=config.device
    )

    task_trajectories = []
    task_stats_collector = defaultdict(list)
    task_renders_list = []

    for i_episode in trange(num_eval_episodes + num_video_episodes, desc=f"Task {task_id} Episodes", leave=False):
        current_episode_trajectory_data = defaultdict(list)
        is_video_episode = i_episode >= num_eval_episodes
        reset_options = {'task_id': task_id}
        reset_options['render_goal'] = is_video_episode

        obs, info = env.reset(options=reset_options)
        task_overall_goal_state = info.get('goal') # Final goal for the entire task from environment
        goal_frame_rendered_from_env = info.get('goal_rendered') # For video rendering

        episode_done = False
        current_episode_step = 0
        rendered_frames_for_this_episode = []
        
        current_plan_observations = None 
        current_subgoal_for_gciql = None   
        current_subgoal_idx_in_plan = -1 

        while (not episode_done) and (current_episode_step < max_epsode_lengths[config.task.env_name]):
            if current_episode_step == 0 or \
               (config.task.replan_every > 0 and current_episode_step % config.task.replan_every == 0):
                                
                normalized_current_obs = normalizer.normalize(obs[:2][None])            
                normalized_task_goal = normalizer.normalize(task_overall_goal_state[:2][None])

                # Create priors for planning
                current_high_level_prior = high_level_prior_template.clone()
                current_high_level_prior[:, 0] = torch.tensor(normalized_current_obs, device=config.device, dtype=torch.float32)
                current_high_level_prior[:, -1] = torch.tensor(normalized_task_goal, device=config.device, dtype=torch.float32)

                # High-level planning
                high_level_traj_normalized, _ = diffusions_models[0].sample(
                    current_high_level_prior.repeat(config.num_candidates, 1, 1),
                    solver=config.solver, n_samples=config.num_candidates,
                    sample_steps=config.sampling_steps, use_ema=config.use_ema,
                    temperature=config.temperature,
                    w_ldg=config.task.w_ldg
                )

                # Prepare for low-level planning
                current_mega_lowlevel_prior = mega_lowlevel_prior_template.clone()
                for i_subgoal in range(H_hi - 1):
                    current_mega_lowlevel_prior[:, i_subgoal, 0] = high_level_traj_normalized[:, i_subgoal]
                    current_mega_lowlevel_prior[:, i_subgoal, -1] = high_level_traj_normalized[:, i_subgoal + 1]
                
                mega_lowlevel_prior_reshaped = current_mega_lowlevel_prior.reshape(
                    config.num_candidates * (H_hi - 1), H_lo, obs_dim)

                # Low-level planning
                mega_segments_normalized, _ = diffusions_models[1].sample(
                    mega_lowlevel_prior_reshaped,
                    solver=config.solver, n_samples=config.num_candidates * (H_hi - 1),
                    sample_steps=config.sampling_steps, use_ema=config.use_ema,
                    temperature=config.temperature,
                )
                mega_segments_normalized = mega_segments_normalized.view(
                    config.num_candidates, H_hi - 1, H_lo, obs_dim)

                # Stitch segments (assuming 0th candidate is best)
                segments_list_normalized = []
                for i_subgoal in range(H_hi - 1):
                    segments_list_normalized.append(
                        mega_segments_normalized[0, i_subgoal, :-1 if i_subgoal < (H_hi - 2) else None, :]
                    )
                full_traj_normalized = torch.cat(segments_list_normalized, dim=0) # Concatenate along time dim
                current_plan_observations = normalizer.unnormalize(full_traj_normalized.cpu().numpy())

                current_subgoal_idx_in_plan = min(config.task.low_horizon, current_plan_observations.shape[0] - 1)
                current_subgoal_for_gciql = current_plan_observations[current_subgoal_idx_in_plan, :]

            elif current_subgoal_for_gciql is not None and current_plan_observations is not None:
                distance_to_subgoal = np.linalg.norm(obs[:2] - current_subgoal_for_gciql)

                if distance_to_subgoal <= config.task.goal_tol: 
                    next_subgoal_potential_idx = current_subgoal_idx_in_plan + config.task.low_horizon
                    print(next_subgoal_potential_idx, current_plan_observations.shape)

                    if next_subgoal_potential_idx < current_plan_observations.shape[0]:
                        current_subgoal_idx_in_plan = next_subgoal_potential_idx
                        current_subgoal_for_gciql = current_plan_observations[current_subgoal_idx_in_plan, :]

                    else: 
                        current_subgoal_idx_in_plan = current_plan_observations.shape[0] - 1
                        current_subgoal_for_gciql = current_plan_observations[current_subgoal_idx_in_plan, :]
                        print(f"Info: Task {task_id}, Ep {i_episode}, Step {current_episode_step} - Reached end of current plan. Aiming for last state.")
            
            gciql_subgoal = obs.copy()
            gciql_subgoal[:2] = current_subgoal_for_gciql[:2]
            action = low_controller.sample_actions(obs, gciql_subgoal, temperature=config.low_eval_temperature)

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_done = info['success']
            current_episode_step += 1

            # Rendering for video episodes
            if is_video_episode and (current_episode_step % video_frame_skip == 0 or episode_done):
                frame = env.render().copy()
                if goal_frame_rendered_from_env is not None:
                    rendered_frames_for_this_episode.append(np.concatenate([goal_frame_rendered_from_env, frame], axis=0))
                else:
                    rendered_frames_for_this_episode.append(frame)
            
            # Store transition data
            if not is_video_episode: # Only store full trajectory data for non-video eval episodes
                transition_payload = dict(
                    observation=obs.copy(), next_observation=next_obs.copy(), action=action.copy(),
                    reward=reward, done=bool(terminated or truncated), info=info.copy()
                )
                add_to(current_episode_trajectory_data, transition_payload)
            
            obs = next_obs

        # End of episode
        if not is_video_episode:
            task_trajectories.append(dict(current_episode_trajectory_data))
            # Use the *last* info from the episode for summary stats
            add_to(task_stats_collector, flatten(info))
        else: # Video episode
            if rendered_frames_for_this_episode:
                task_renders_list.append(np.array(rendered_frames_for_this_episode))

    # Aggregate statistics for this task
    aggregated_stats_for_this_task = {}
    for k, v_list in task_stats_collector.items():
        numeric_vals = [x for x in v_list if isinstance(x, (int, float, np.number))]
        if numeric_vals:
            aggregated_stats_for_this_task[k] = np.mean(numeric_vals)
        elif v_list: # Handle non-numeric summary data if any (e.g. list of strings)
            aggregated_stats_for_this_task[k] = v_list

    return aggregated_stats_for_this_task, task_trajectories, task_renders_list, current_plan_observations, current_subgoal_for_gciql



@hydra.main(config_path="../../configs/scots/ogbench", config_name="ogbench", version_base=None)
def pipeline(args):

    wandb.init(
        config=OmegaConf.to_container(args, resolve=True),
        project='scots-ogbench',
        group=args.group,
        name=f"scots_{args.task.env_name}",
        id=str(uuid.uuid4()), # Generate unique ID
        dir=tempfile.mkdtemp(), # Use temp dir for wandb files
    )

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    stitcher_path = f'results/stitcher_ogbench_H{args.task.horizons[1]}/{args.task.env_name}/'
    
    summary_dir = f'results/{args.pipeline_name}/{args.task.env_name}/'
    summary_filename = os.path.join(summary_dir, f"hyp_sweep_summary_{args.group}.csv")

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    planning_horizons = args.task.horizons
    # ========================== Level Setup ==========================
    n_levels = len(planning_horizons)
    temporal_horizons = [planning_horizons[-1] for _ in range(n_levels)]
    for i in range(n_levels - 1):
        temporal_horizons[-2 - i] = (planning_horizons[-2 - i] - 1) * (temporal_horizons[-1 - i] - 1) + 1

    # ---------------------- Create Dataset ----------------------
    env, dataset, _ = ogbench.make_env_and_datasets(
        args.task.env_name,
        compact_dataset=True,
    )
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    ob_dtype = np.uint8 if ('visual' in args.task.env_name or 'powderworld' in args.task.env_name) else np.float32
    action_dtype = np.int32 if 'powderworld' in args.task.env_name else np.float32
    aug_dataset = load_dataset(
        os.path.join(stitcher_path, f'{args.task.env_name}'+'_augmented.npz'),
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=True,
        add_info=False,
    )

    # utilize augmented data
    if 'navigate' in args.task.env_name:
        dataset = {
            key: np.concatenate([dataset[key], aug_dataset[key]], axis=0)
            for key in dataset.keys()
        }
    else:
        dataset = aug_dataset
    
    dataset = MultiHorizonOGBenchDataset(
        dataset, 
        horizons=temporal_horizons,
        only_xy=True,
        with_ball=False
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    goal_dim = 2

    # =========================== Model Setup ==========================
    fix_masks = [torch.zeros((h, goal_dim)) for h in planning_horizons]
    loss_weights = [torch.ones((h, goal_dim)) for h in planning_horizons]
    for i in range(n_levels):
        fix_idx = [0, -1] # both high, low goal-conditioned
        fix_masks[i][fix_idx, :] = 1.

    nn_diffusions = [
        DiT1d(
            goal_dim, emb_dim=128,
        d_model=256, n_heads=256//64, depth=8, timestep_emb_type="fourier")
        for _ in range(n_levels)]
    
    diffusions = [
        ContinuousDiffusionSDEEX(
            nn_diffusions[i], nn_condition=None,
            fix_mask=fix_masks[i], loss_weight=loss_weights[i], classifier=None, ema_rate=0.9999,
            device=args.device, predict_noise=True, noise_schedule="linear")
        for i in range(n_levels)]
    

    # ---------------------- Training ----------------------
    if args.mode == "train":

        progress_bar = tqdm(total=args.diffusion_gradient_steps, desc="Training Progress")

        diffusion_lr_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(diffusions[i].optimizer, args.diffusion_gradient_steps)
            for i in range(n_levels)]

        for diffusion in diffusions:
            diffusion.train()

        n_gradient_step = 0
        log = dict.fromkeys(
            [f"diffusion_loss{i}" for i in range(n_levels)], 0.)

        for batch in loop_dataloader(dataloader):
            for i in range(n_levels):
                batch_data = batch[i]["data"]

                obs = batch_data["obs"]["state"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(
                    args.device)

                if args.use_mask_strategy:
                    mask_i = torch.randint(0, obs.shape[1]-1, (1,)).item()
                    obs[:, mask_i+1:, :] = obs[:, mask_i:mask_i+1, :]

                log[f"diffusion_loss{i}"] += diffusions[i].update(obs)["loss"]
                diffusion_lr_schedulers[i].step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                print(log)
                wandb.log(log, step=n_gradient_step + 1)
                log = dict.fromkeys(
                    [f"diffusion_loss{i}" for i in range(n_levels)], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                for i in range(n_levels):
                    diffusions[i].save(save_path + f'diffusion{i}_ckpt_{n_gradient_step + 1}.pt')
                    diffusions[i].save(save_path + f'diffusion{i}_ckpt_latest.pt')

            n_gradient_step += 1
            progress_bar.update(1)

            if n_gradient_step > args.diffusion_gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        normalizer = dataset.get_normalizer()

        for i in range(n_levels):
            diffusions[i].load(
                save_path + f'{"diffusion"}{i}_ckpt_{args.diffusion_ckpt}.pt')
            diffusions[i].eval()
            print(save_path + f'{"diffusion"}{i}_ckpt_{args.diffusion_ckpt}.pt')

        if args.task.low_controller == 'gciql':
            low_controller_save_path = f'results/GCIQL/{args.task.env_name}/'
            low_controller = GCIQL(obs_dim, act_dim, device=args.device)
            low_controller.load(low_controller_save_path + f'gciql_ckpt_latest.pt')
            print(low_controller_save_path + f'gciql_ckpt_latest.pt')
        elif args.task.low_controller == 'crl':
            low_controller_save_path = f'results/CRL/{args.task.env_name}/'
            low_controller = CRL(obs_dim, act_dim, device=args.device)
            low_controller.load(low_controller_save_path + f'crl_ckpt_latest.pt')
            print(low_controller_save_path + f'crl_ckpt_latest.pt')
        else:
            raise NotImplementedError
        low_controller.eval()

        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = len(task_infos)
        for task_id in trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]['task_name']
            eval_info, trajs, cur_renders, cur_plan, cur_subgoal = evaluate(
                diffusions_models=diffusions,
                low_controller=low_controller,
                env=env,
                normalizer=normalizer,
                task_id=task_id,
                planning_horizons=planning_horizons, # from args.task.horizons
                obs_dim=goal_dim, # TODO: make code clear and readable
                config=args, # The main Hydra args object
                num_eval_episodes=args.eval_episodes,
                num_video_episodes=args.video_episodes,
                video_frame_skip=args.frame_skip,
            )
            renders.extend(cur_renders)
            metric_names = ['success']
            eval_metrics.update(
                {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

        if args.video_episodes > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics['video'] = video

        wandb.log(eval_metrics, step=i)

        # Define the header (should match the order of data written)
        fieldnames = [
            'seed', 'low_controller', 'goal_tol', 'replan_every', 'low_horizon', 'w_ldg',
            'overall_success', # Add other key metrics you want to compare
        ]

        # Data for the current run
        current_run_data = {
            'seed': args.seed,
            'low_controller': args.task.low_controller,
            'goal_tol': args.task.goal_tol, # Make sure goal_tol is accessible here (add to args?)
            'replan_every': args.task.replan_every,
            'low_horizon': args.task.low_horizon, # Make sure gciql_horizon is accessible (add to args?)
            'w_ldg': args.task.w_ldg, # Make sure w_ldg is accessible (add to args?)
            'overall_success': eval_metrics['evaluation/overall_success'],
        }

        # Check if file exists to write header only once
        write_header = not os.path.exists(summary_filename)

        with open(summary_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader() # Write header if file is new

            # Write the data for the current run
            writer.writerow(current_run_data)

        print(f"Appended results to summary file: {summary_filename}")


    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
if __name__ == "__main__":
    pipeline()