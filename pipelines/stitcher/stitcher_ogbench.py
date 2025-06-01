import os
import ogbench
import hydra, wandb, uuid, tempfile
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.invdynamic import FancyMlpInvDynamic
from cleandiffuser.utils import report_parameters

from cleandiffuser_ex.utils import set_seed
from cleandiffuser_ex.hilp import HILP
from cleandiffuser_ex.stitch import stitch_single_rollout
from cleandiffuser_ex.faiss_index_wrapper import FaissIndexWrapper
from cleandiffuser_ex.diffusion import ContinuousDiffusionSDEEX
from cleandiffuser_ex.dataset.ogbench_dataset import OGBenchDataset


@hydra.main(config_path="../../configs/stitcher/ogbench", config_name="ogbench", version_base=None)
def pipeline(args):

    wandb.init(
        config=OmegaConf.to_container(args, resolve=True),
        project='scots-ogbench',
        group=args.group,
        name=f"stitcher_H{args.task.horizon}_{args.task.env_name}",
        id=str(uuid.uuid4()), # Generate unique ID
        dir=tempfile.mkdtemp(), # Use temp dir for wandb files
    )

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}_H{args.task.horizon}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env, dataset, _ = ogbench.make_env_and_datasets(
        args.task.env_name,
        compact_dataset=True,
    )

    dataset = OGBenchDataset(dataset, horizon=args.task.horizon)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = DiT1d(
        obs_dim, emb_dim=128,
        d_model=256, n_heads=256//64, depth=8, timestep_emb_type="fourier")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim))
    fix_mask[[0, -1]] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim))

    # --------------- Diffusion Model --------------------
    agent = ContinuousDiffusionSDEEX(
        nn_diffusion, nn_condition=None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=None, ema_rate=args.ema_rate,
        device=args.device, predict_noise=True, noise_schedule="linear")

    # ---------------------- Training ----------------------
    if args.mode == "train":

        progress_bar = tqdm(total=args.diffusion_gradient_steps, desc="Training Progress")

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            if args.use_mask_strategy:
                mask_i = torch.randint(0, obs.shape[1]-1, (1,)).item()
                obs[:, mask_i+1:, :] = obs[:, mask_i:mask_i+1, :]

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(obs)['loss']
            diffusion_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                print(log)
                wandb.log(log, step=n_gradient_step + 1)
                log = {"avg_loss_diffusion": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"stitcher_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"stitcher_ckpt_latest.pt")

            n_gradient_step += 1
            progress_bar.update(1)

            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    elif args.mode == "generate_data":

        invdyn_save_path = f'results/invdyn_ogbench/{args.task.env_name}/'
        hilp_save_path = f'results/HILP/{args.task.env_name}/'

        agent.load(save_path + "stitcher_ckpt_latest.pt")
        agent.eval()

        normalizer = dataset.get_normalizer()
        actual_dataset_size = min(args.dataset_size_limit, len(dataset))

        print("Loading HILP model...")
        hilp = HILP(obs_dim, args.latent_dim, args.device) # latent_dim=32 가정
        latent_dim = args.latent_dim
        hilp.load(hilp_save_path + "hilp_ckpt_latest.pt")
        hilp.eval()

        print("Loading invdyn model...")
        invdyn = FancyMlpInvDynamic(obs_dim, act_dim, 256, nn.Tanh(), add_dropout=True, device=args.device)
        invdyn.load(invdyn_save_path + f'invdyn_ckpt_latest.pt')
        invdyn.eval()

        traj_dataset = np.zeros((actual_dataset_size, args.task.horizon, obs_dim), dtype=np.float32)
        gen_dl = DataLoader(dataset, batch_size=5000, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
        ptr = 0
        with tqdm(total=actual_dataset_size, desc="Filtering Data", leave=False) as pbar:
            for batch in gen_dl:
                obs_torch = batch["obs"]["state"]; bs = obs_torch.shape[0]
                if bs == 0: continue
                unnorm = normalizer.unnormalize(obs_torch.cpu().numpy())
                start_obs, end_obs = unnorm[:, 0, :], unnorm[:, -1, :]

                start_xy = start_obs[:, :2]    # Shape: [bs, 2]
                end_xy   = end_obs[:, :2]   # Shape: [bs, 2]

                d_star = np.linalg.norm(end_xy - start_xy, axis=1)

                valid_idx = np.where(d_star >= args.task.distance_threshold)[0]
                if valid_idx.size == 0: continue
                num_to_take = min(len(valid_idx), actual_dataset_size - ptr)
                indices_to_take = valid_idx[:num_to_take]
                traj_dataset[ptr : ptr + num_to_take] = unnorm[indices_to_take]
                ptr += num_to_take; pbar.update(num_to_take)
                if ptr >= actual_dataset_size: break
        if ptr < actual_dataset_size: traj_dataset = traj_dataset[:ptr]
        print(f"Filtered dataset created. Size: {traj_dataset.shape[0]}")

        print("Computing latent dataset for Faiss index (using start states)...")
        latent_dataset = np.zeros((traj_dataset.shape[0], latent_dim), dtype=np.float32)
        batch_size_phi = 1024
        for i in tqdm(range(0, traj_dataset.shape[0], batch_size_phi), desc="Computing Latents"):
            end_idx = min(i + batch_size_phi, traj_dataset.shape[0])
            obs_batch = traj_dataset[i:end_idx, 0, :] 
            latent_dataset[i:end_idx] = hilp.get_phi(obs_batch)
        print("Building Faiss index...")
        faiss_wrapper_latent = FaissIndexWrapper(
            similarity_metric="l2", data=latent_dataset, device=args.device
        )
        print(f"Faiss index built with {len(faiss_wrapper_latent)} vectors.")

        print(f"\nStarting data generation: {args.num_episodes_to_generate} episodes...")
        np.random.seed(args.seed) 
        generated_trajs = []

        target_dim = latent_dim if args.use_hilp_for_rollout else obs_dim
        print(f"Using {'LATENT' if args.use_hilp_for_rollout else 'OBSERVATION'} space (dim={target_dim}) for progress direction.")
        print(f"Novelty calculation is based on {'LATENT' if args.use_hilp_for_rollout else 'OBSERVATION'} space history.")

        total_train_steps = args.num_episodes_to_generate * args.task.max_episode_steps
        num_train_episodes = args.num_episodes_to_generate
        num_val_episodes = args.num_episodes_to_generate // 10
        total_eps = num_train_episodes + num_val_episodes
        for i in tqdm(range(total_eps), desc="Generating Stitched Episodes"):
            start_idx = np.random.randint(0, len(traj_dataset))
            initial_obs_ep = traj_dataset[start_idx:start_idx+1, 0, :]

            z_direction = np.random.randn(target_dim).astype(np.float32)
            z_norm = np.linalg.norm(z_direction)
            z_direction = z_direction / z_norm

            generated_traj = stitch_single_rollout(
                initial_obs=initial_obs_ep,
                z_direction=z_direction,
                use_hilp=args.use_hilp_for_rollout,
                num_steps=args.task.max_episode_steps // (args.task.horizon - 1),
                hilp_model=hilp,
                faiss_index=faiss_wrapper_latent,
                full_traj_dataset=traj_dataset,
                k_neighbors=args.k_neighbors_rollout,
                k_density=args.k_density,
                alpha=args.alpha,
                beta=args.beta,
                pbar_desc=f"Ep {i+1}"
            )

            generated_trajs.append(generated_traj.astype(np.float32))

        generated_trajs = np.array(generated_trajs)
        stitched_trajs = generated_trajs.copy()

        # refining using stitcher
        batch_size_stitcher = 100
        for i in tqdm(range(0, generated_trajs.shape[0], batch_size_stitcher), desc="Smoothing Trajs"):
            end_idx = min(i + batch_size_stitcher, generated_trajs.shape[0])
            batch_generated_trajs = generated_trajs[i:end_idx]
            for segment_t in range(0, generated_trajs.shape[1] - 1, args.task.horizon - 1):
                segment = batch_generated_trajs[:, segment_t:segment_t + args.task.horizon]
                segment = torch.tensor(normalizer.normalize(segment), device=args.device, dtype=torch.float32)
            
                prior = torch.zeros((segment.shape[0], args.task.horizon, obs_dim), device=args.device)
                prior[:, 0, :obs_dim] = segment[:, 0]
                prior[:, -1, :obs_dim] = segment[:, -1]

                stitched_traj, log = agent.sample(
                    prior,
                    solver=args.solver,
                    n_samples=segment.shape[0],
                    use_ema=args.use_ema, 
                    temperature=args.temperature,
                    sample_steps=args.sampling_steps, 
                    preserve_history=False
                )

                stitched_trajs[i:end_idx, segment_t:segment_t + args.task.horizon] = normalizer.unnormalize(stitched_traj.cpu().numpy())

        batch_size_invdyn = 100 
        all_actions_list = []
        for i in tqdm(range(0, stitched_trajs.shape[0], batch_size_invdyn), desc="Smoothing Trajs"):
            end_idx = min(i + batch_size_invdyn, stitched_trajs.shape[0])
            current_batch_trajs_np = stitched_trajs[i:end_idx]

            obs_data_np = current_batch_trajs_np

            # Shape: (current_batch_size, L, D)
            next_obs_part1_np = current_batch_trajs_np[:, 1:, :]
            
            # Shape: (current_batch_size, 1, D)
            last_obs_repeated_np = current_batch_trajs_np[:, -1:, :]
            next_obs_data_np = np.concatenate((next_obs_part1_np, last_obs_repeated_np), axis=1)

            obs = torch.tensor(normalizer.normalize(obs_data_np), device=args.device, dtype=torch.float32)
            next_obs = torch.tensor(normalizer.normalize(next_obs_data_np), device=args.device, dtype=torch.float32)
            
            actions = invdyn(obs, next_obs)
            all_actions_list.append(actions.detach().cpu().numpy())

        final_actions_array = np.concatenate(all_actions_list, axis=0)

        terminals = np.zeros((stitched_trajs.shape[0], stitched_trajs.shape[1]), dtype=np.float32)
        terminals[:, -1] = 1

        dataset = defaultdict(list)
        dataset['observations'] = stitched_trajs.reshape(-1, obs_dim)
        dataset['actions'] = final_actions_array.reshape(-1, act_dim)
        dataset['terminals'] = terminals.reshape(-1)

        # Split the dataset into training and validation sets.
        train_dataset = {}
        val_dataset = {}
        for k, v in dataset.items():
            if 'observations' in k and v[0].dtype == np.uint8:
                dtype = np.uint8
            elif k == 'terminals':
                dtype = bool
            else:
                dtype = np.float32
            train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
            val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

        train_path = save_path + f"{args.task.env_name}_augmented.npz"
        val_path = train_path.replace('.npz', '-val.npz')

        for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
            np.savez_compressed(path, **dataset)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()