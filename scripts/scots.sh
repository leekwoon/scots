#!/usr/bin/env bash
set -e

export MUJOCO_GL=egl


python pipelines/scots/scots_ogbench.py task=pointmaze-medium-stitch-v0 
python pipelines/scots/scots_ogbench.py task=pointmaze-large-stitch-v0 
python pipelines/scots/scots_ogbench.py task=pointmaze-giant-stitch-v0 
python pipelines/scots/scots_ogbench.py task=antmaze-medium-stitch-v0 
python pipelines/scots/scots_ogbench.py task=antmaze-large-stitch-v0 
python pipelines/scots/scots_ogbench.py task=antmaze-giant-stitch-v0 
python pipelines/scots/scots_ogbench.py task=antmaze-medium-explore-v0 
python pipelines/scots/scots_ogbench.py task=antmaze-large-explore-v0 

