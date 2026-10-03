#!/usr/bin/env bash
# Configuration, input-file, GPU, and model preflight checks.

check_launch_config() {
    if [[ "${NUM_GPUS}" -ne "${#GPU_IDS[@]}" ]]; then
        echo "NUM_GPUS=${NUM_GPUS} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
        exit 2
    fi

    if ((TRAIN_PROMPT_BATCH_SIZE % NUM_GPUS != 0)); then
        echo "TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}" >&2
        exit 2
    fi
    if ((PPO_PROMPT_MINI_BATCH_SIZE % NUM_GPUS != 0)); then
        echo "PPO_MINI_BATCH_SIZE=${PPO_PROMPT_MINI_BATCH_SIZE} trajectories must be divisible by NUM_GPUS=${NUM_GPUS}" >&2
        exit 2
    fi

    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Python executable is missing: ${PYTHON_BIN}" >&2
        exit 2
    fi
}

check_training_inputs() {
    for data_file in "${TRAIN_FILES[@]}" "${VAL_FILES[@]}"; do
        if [[ ! -f "${data_file}" ]]; then
            echo "Missing dataset: ${data_file}" >&2
            exit 2
        fi
    done

    if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
        for gpu_id in "${GPU_IDS[@]}"; do
            free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}")
            if [[ "${free_mib}" -lt "${MIN_FREE_GPU_MIB}" ]]; then
                echo "GPU ${gpu_id} has only ${free_mib} MiB free; need ${MIN_FREE_GPU_MIB} MiB" >&2
                exit 3
            fi
        done
    fi
}

check_model() {
    if [[ "${SKIP_MODEL_PREFLIGHT:-0}" != "1" ]]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/lib/check_model.py" "${MODEL_PATH}" "${MAX_MODEL_LENGTH}"
    fi
}
