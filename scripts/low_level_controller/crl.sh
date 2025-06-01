set -e

export MUJOCO_GL=egl

python main_aug.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.03
python main_aug.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.03
python main_aug.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.03 --agent.discount=0.995
python main_aug.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
python main_aug.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
python main_aug.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
python main_aug.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.003
python main_aug.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/crl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.003
