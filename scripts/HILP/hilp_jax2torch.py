import os
import glob
import ogbench
import ogbench_ex
import pyrallis
import dataclasses
from matplotlib import pyplot as plt

from cleandiffuser_ex.hilp import HILP, load_hilp_jax_checkpoint_to_pytorch
from cleandiffuser_ex.plot_utils import generate_tsne_visualization


@dataclasses.dataclass
class Config:
    env_name: str = "pointmaze-large-stitch-v0"  # OpenAI gym environment name
    skill_dim: int = 32      # Dimension of the phi representation
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
    hilp = HILP(obs_dim, config.skill_dim, "cuda") 

    print(config.save_dir)
    candidates = glob.glob(config.save_dir)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{config.restore_epoch}.pkl'

    load_hilp_jax_checkpoint_to_pytorch(restore_path, hilp)
    hilp.save(os.path.join(config.save_dir, f"hilp_ckpt_{config.restore_epoch}.pt"))
    hilp.save(os.path.join(config.save_dir, f"hilp_ckpt_latest.pt"))

    fig, ax = plt.subplots(figsize=(6, 6))
    generate_tsne_visualization(
        env, hilp, dataset, ax=ax
    )
    fig.savefig(os.path.join(config.save_dir, 'tsne.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()