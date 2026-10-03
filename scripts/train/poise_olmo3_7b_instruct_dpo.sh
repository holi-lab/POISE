#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export POISE_MODEL_PROFILE=olmo3_7b
export MODEL_PATH=${MODEL_PATH:-allenai/Olmo-3-7B-Instruct-DPO}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-poise-olmo3-7b-instruct-dpo}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}
exec "${SCRIPT_DIR}/_poise.sh" "$@"
