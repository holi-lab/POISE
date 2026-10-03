#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# POISE settings shared by the Qwen3 and OLMo3 launchers.

export TRAIN_SCOPE=${TRAIN_SCOPE:-full}
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-poise}
export TRAIN_ENTRYPOINT=poise.main
export TRAIN_CONFIG_NAME=poise_fsdp
export TRAIN_ALGORITHM_LABEL=poise

# POISE computes cross-rollout advantages inside RayPOISETrainer. GRPO remains
# the base estimator label only for shared PPO configuration compatibility, so
# no critic is instantiated.

export ADV_ESTIMATOR=grpo
export ROLLOUTS_PER_PROMPT=2
export TOTAL_STEPS=${TOTAL_STEPS:-200}
export DATA_SHUFFLE=${DATA_SHUFFLE:-True}
export TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-256}
export GEN_PROMPT_BATCH_SIZE=${GEN_PROMPT_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

# Initial bundles are optional when starting from the default RLOO bootstrap.
POISE_ARTIFACT_DIR="${REPO_ROOT}/artifacts"
POISE_CONFIG_DIR="${REPO_ROOT}/src/poise/config"
SHARED_VAL_DIR=${VAL_DATA_DIR:-"${REPO_ROOT}/data/online_eval"}
POISE_VAL_DIR="${SHARED_VAL_DIR}/poise"
export POISE_MATH_ESTIMATOR_PATH=${POISE_MATH_ESTIMATOR_PATH:-"${POISE_ARTIFACT_DIR}/initial_estimator_math_${POISE_MODEL_PROFILE}.joblib"}
export POISE_CODE_ESTIMATOR_PATH=${POISE_CODE_ESTIMATOR_PATH:-"${POISE_ARTIFACT_DIR}/initial_estimator_code_${POISE_MODEL_PROFILE}.joblib"}
export POISE_OTHER_ESTIMATOR_PATH=${POISE_OTHER_ESTIMATOR_PATH:-"${POISE_ARTIFACT_DIR}/initial_estimator_others_${POISE_MODEL_PROFILE}.joblib"}
export POISE_FEATURE_BUILDER_CONFIG=${POISE_FEATURE_BUILDER_CONFIG:-"${POISE_CONFIG_DIR}/feature_builder_config_${POISE_MODEL_PROFILE}.json"}
export POISE_ESTIMATOR_FIT_CONFIG=${POISE_ESTIMATOR_FIT_CONFIG:-"${POISE_CONFIG_DIR}/estimator_fit_config_${POISE_MODEL_PROFILE}.json"}
export POISE_ESTIMATOR_WARMUP_STEPS=${POISE_ESTIMATOR_WARMUP_STEPS:-8}
export POISE_ESTIMATOR_SAVE_FREQ=${POISE_ESTIMATOR_SAVE_FREQ:-10}
export POISE_ESTIMATOR_ONLINE_OUTPUT_DIR=${POISE_ESTIMATOR_ONLINE_OUTPUT_DIR:-null}
export POISE_MATH_BUFFER_MAX_ROWS=${POISE_MATH_BUFFER_MAX_ROWS:-4096}
export POISE_CODE_BUFFER_MAX_ROWS=${POISE_CODE_BUFFER_MAX_ROWS:-3072}
export POISE_OTHER_BUFFER_MAX_ROWS=${POISE_OTHER_BUFFER_MAX_ROWS:-3072}
export POISE_ESTIMATOR_EVAL_ON_VAL=${POISE_ESTIMATOR_EVAL_ON_VAL:-false}

# Estimator validation needs multiple responses for every prompt.
case "${POISE_ESTIMATOR_EVAL_ON_VAL,,}" in
    true)
        POISE_DEFAULT_VALIDATION_ROLLOUTS=8
        POISE_DEFAULT_AIME_VAL_FILE="${POISE_VAL_DIR}/math__aime_unique_30.parquet"
        POISE_DEFAULT_AMC_VAL_FILE="${POISE_VAL_DIR}/math__amc_unique_83.parquet"
        ;;
    false)
        POISE_DEFAULT_VALIDATION_ROLLOUTS=1
        POISE_DEFAULT_AIME_VAL_FILE="${SHARED_VAL_DIR}/math__aime_repeated_8x_240.parquet"
        POISE_DEFAULT_AMC_VAL_FILE="${SHARED_VAL_DIR}/math__amc_repeated_4x_332.parquet"
        ;;
    *)
        echo "POISE_ESTIMATOR_EVAL_ON_VAL must be True or False" >&2
        exit 2
        ;;
esac
export POISE_VALIDATION_ROLLOUTS=${POISE_VALIDATION_ROLLOUTS:-${POISE_DEFAULT_VALIDATION_ROLLOUTS}}
export POISE_AIME_VAL_FILE=${POISE_AIME_VAL_FILE:-${POISE_DEFAULT_AIME_VAL_FILE}}
export POISE_AMC_VAL_FILE=${POISE_AMC_VAL_FILE:-${POISE_DEFAULT_AMC_VAL_FILE}}
export POISE_BOOTSTRAP_FROM_SCRATCH=${POISE_BOOTSTRAP_FROM_SCRATCH:-True}

# Bootstrap uses one fixed prompt batch and RLOO advantages per step.
if [[ "${POISE_BOOTSTRAP_FROM_SCRATCH}" == "True" || "${POISE_BOOTSTRAP_FROM_SCRATCH}" == "true" ]]; then
    export POISE_BOOTSTRAP_STEPS=${POISE_BOOTSTRAP_STEPS:-16}
else
    export POISE_BOOTSTRAP_STEPS=${POISE_BOOTSTRAP_STEPS:-0}
fi

export VERL_PROCESS_GROUP_TIMEOUT_SECONDS=${VERL_PROCESS_GROUP_TIMEOUT_SECONDS:-10800}

if [[ "${CONFIG_ONLY:-0}" != "1" ]]; then
    required_files=(
        "${POISE_FEATURE_BUILDER_CONFIG}"
        "${POISE_ESTIMATOR_FIT_CONFIG}"
    )
    estimator_eval_on_val=0
    if [[ "${POISE_ESTIMATOR_EVAL_ON_VAL}" == "True" || "${POISE_ESTIMATOR_EVAL_ON_VAL}" == "true" ]]; then
        estimator_eval_on_val=1
        required_files+=(
            "${POISE_AIME_VAL_FILE}"
            "${POISE_AMC_VAL_FILE}"
        )
    fi
    initial_estimator_needed=0
    if [[ "${POISE_BOOTSTRAP_FROM_SCRATCH}" != "True" && "${POISE_BOOTSTRAP_FROM_SCRATCH}" != "true" ]]; then
        initial_estimator_needed=1
    elif ((estimator_eval_on_val)) &&
         [[ "${VAL_BEFORE_TRAIN:-True}" == "True" || "${VAL_BEFORE_TRAIN:-True}" == "true" ]]; then
        initial_estimator_needed=1
    fi
    if ((initial_estimator_needed)); then
        required_files+=(
            "${POISE_MATH_ESTIMATOR_PATH}"
            "${POISE_CODE_ESTIMATOR_PATH}"
            "${POISE_OTHER_ESTIMATOR_PATH}"
        )
    fi
    for required_file in "${required_files[@]}"; do
        if [[ ! -f "${required_file}" ]]; then
            echo "Missing POISE estimator input: ${required_file}" >&2
            exit 2
        fi
    done
fi

exec "${SCRIPT_DIR}/_launch.sh" "$@"
