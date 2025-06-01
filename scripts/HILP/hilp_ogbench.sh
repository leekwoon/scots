#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- Configuration ---
# Define the actual directory where the script is located (Robust way)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT="${SCRIPT_DIR}/hilp_ogbench.py"

# Define common parameters
SAVE_INTERVAL=1000000
RESULTS_BASE_DIR="${SCRIPT_DIR}/../../results"

# Define environments and subgoal steps separately
ENVS=(
    "pointmaze-medium-stitch-v0" "pointmaze-large-stitch-v0" "pointmaze-giant-stitch-v0"
    "antmaze-medium-stitch-v0" "antmaze-large-stitch-v0" "antmaze-giant-stitch-v0"
    "antmaze-medium-explore-v0" "antmaze-large-explore-v0"
)

# --- Execution ---
# Outer loop for environments
for env_name in "${ENVS[@]}"; do
    if [[ "$env_name" == *giant* || "$env_name" == *humanoidmaze* ]]; then
        DISCOUNT=0.995
    else
        DISCOUNT=0.99
    fi

    # Construct the save directory path
    save_dir_path="${RESULTS_BASE_DIR}/HILP/${env_name}"

    # Execute the python script with the specified parameters
    python "$PYTHON_SCRIPT" \
        --env_name "$env_name" \
        --discount "$DISCOUNT" \
        --save_interval "$SAVE_INTERVAL" \
        --save_dir "$save_dir_path" \

    echo "Finished run for Env=$env_name, Steps=$steps."
    echo # Add a blank line for readability
    
done # End outer loop (environments)

echo "============================="
echo "All training runs completed successfully."


