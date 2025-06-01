import os
import ogbench
import hydra, wandb, uuid, tempfile
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.invdynamic import FancyMlpInvDynamic

from cleandiffuser_ex.utils import set_seed
from cleandiffuser_ex.dataset.ogbench_dataset import OGBenchDataset


@hydra.main(config_path="../../configs/invdyn/ogbench", config_name="ogbench", version_base=None)
def pipeline(args):

    wandb.init(
        config=OmegaConf.to_container(args, resolve=True),
        project='scots-ogbench',
        group=args.group,
        name=f"invdyn_H{args.task.horizon}_{args.task.env_name}",
        id=str(uuid.uuid4()), 
        dir=tempfile.mkdtemp(), 
    )

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
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

    # --------------- Inverse Dynamic -------------------
    invdyn = FancyMlpInvDynamic(obs_dim, act_dim, 256, nn.Tanh(), add_dropout=True, device=args.device)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        progress_bar = tqdm(total=args.invdyn_gradient_steps, desc="Training Progress")

        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)

        invdyn.train()

        n_gradient_step = 0
        log = {"avg_loss_invdyn": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)

            # ----------- Gradient Step ------------
            log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
            invdyn_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_invdyn"] /= args.log_interval
                print(log)
                wandb.log(log, step=n_gradient_step + 1)
                log = {"avg_loss_invdyn": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
                invdyn.save(save_path + f"invdyn_ckpt_latest.pt")

            n_gradient_step += 1
            progress_bar.update(1)

            if n_gradient_step >= args.invdyn_gradient_steps:
                break

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()