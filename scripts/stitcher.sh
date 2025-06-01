#!/usr/bin/env bash
set -e

export MUJOCO_GL=egl

# train stitcher
python pipelines/stitcher/stitcher_ogbench.py mode=train task=pointmaze-medium-stitch-v0 diffusion_gradient_steps=200000
python pipelines/stitcher/stitcher_ogbench.py mode=train task=pointmaze-large-stitch-v0 diffusion_gradient_steps=200000
python pipelines/stitcher/stitcher_ogbench.py mode=train task=pointmaze-giant-stitch-v0 diffusion_gradient_steps=200000
python pipelines/stitcher/stitcher_ogbench.py mode=train task=antmaze-medium-stitch-v0
python pipelines/stitcher/stitcher_ogbench.py mode=train task=antmaze-large-stitch-v0
python pipelines/stitcher/stitcher_ogbench.py mode=train task=antmaze-giant-stitch-v0
python pipelines/stitcher/stitcher_ogbench.py mode=train task=antmaze-medium-explore-v0
python pipelines/stitcher/stitcher_ogbench.py mode=train task=antmaze-large-explore-v0

# augment
python pipelines/stitcher/stitcher_ogbench.py mode=generate_data task=pointmaze-medium-stitch-v0 num_episodes_to_generate=5000
python pipelines/stitcher/stitcher_ogbench.py mode=generate_data task=pointmaze-large-stitch-v0 num_episodes_to_generate=5000
python pipelines/stitcher/stitcher_ogbench.py mode=generate_data task=pointmaze-giant-stitch-v0 num_episodes_to_generate=5000
python pipelines/stitcher/stitcher_ogbench.py mode=generate_data task=antmaze-medium-stitch-v0 num_episodes_to_generate=5000
python pipelines/stitcher/stitcher_ogbench.py mode=generate_data task=antmaze-large-stitch-v0 num_episodes_to_generate=5000
python pipelines/stitcher/stitcher_ogbench.py mode=generate_data task=antmaze-giant-stitch-v0 num_episodes_to_generate=5000
python pipelines/stitcher/stitcher_ogbench.py mode=generate_data task=antmaze-medium-explore-v0 num_episodes_to_generate=5000
python pipelines/stitcher/stitcher_ogbench.py mode=generate_data task=antmaze-large-explore-v0 num_episodes_to_generate=5000