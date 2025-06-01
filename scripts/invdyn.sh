#!/usr/bin/env bash
set -e

export MUJOCO_GL=egl


ENVS=(
    "pointmaze-medium-stitch-v0"  "pointmaze-large-stitch-v0"  "pointmaze-giant-stitch-v0"
    "antmaze-medium-stitch-v0"    "antmaze-large-stitch-v0"    "antmaze-giant-stitch-v0"
    "antmaze-medium-explore-v0"   "antmaze-large-explore-v0"
)


for env in "${ENVS[@]}"; do
    echo -e "\n▶ Running ${env} (horizon=${horizon})"
    python pipelines/invdyn/invdyn_ogbench.py task="$env"
done
