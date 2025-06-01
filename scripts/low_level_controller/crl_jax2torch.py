import os
os.environ['MUJOCO_GL'] = 'egl'
import glob
import ogbench
import ogbench_ex
import pyrallis
import dataclasses

from cleandiffuser_ex.crl import CRL, load_crl_jax_checkpoint_to_pytorch


@dataclasses.dataclass
class Config:
    env_name: str = "pointmaze-large-navigate-v0"  # OpenAI gym environment name
    goal_rep_dim: int = 10      # Dimension of the phi representation
    save_dir: str = 'exp/'
    restore_epoch: int = 1000000 


@pyrallis.wrap()
def main(config: Config):
    if 'checker' in config.env_name:
        env, dataset, _ = ogbench_ex.make_env_and_datasets(
            config.env_name,
            compact_dataset=False,
        )
    else:
        env, dataset, _ = ogbench.make_env_and_datasets(
            config.env_name,
            compact_dataset=False,
        )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    crl = CRL(obs_dim, act_dim, device="cuda") 

    candidates = glob.glob(f'./exp/crl-{config.env_name}')
    # candidates = glob.glob(config.save_dir)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{config.restore_epoch}.pkl'

    load_crl_jax_checkpoint_to_pytorch(restore_path, crl)

    if os.path.exists(config.save_dir) is False:
        os.makedirs(config.save_dir)

    crl.save(os.path.join(config.save_dir, f"crl_ckpt_{config.restore_epoch}.pt"))
    crl.save(os.path.join(config.save_dir, f"crl_ckpt_latest.pt"))


if __name__ == '__main__':
    main()
