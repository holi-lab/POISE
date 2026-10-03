#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export POISE_MODEL_PROFILE=qwen3_4b
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-poise-qwen3-4b}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}
exec "${SCRIPT_DIR}/_poise.sh" "$@"
