#!/usr/bin/env bash
# SandboxFusion health checks and owned STEM verifier lifecycle.

configure_reward_services() {
    # For full-domain training, start the 1.5B STEM judge on the first training GPU
    # unless an external STEM_LLM_JUDGE_URL is already provided.
    AUTO_START_STEM_VERIFIER=${AUTO_START_STEM_VERIFIER:-1}
    STEM_VERIFIER_GPU=${STEM_VERIFIER_GPU:-${GPU_IDS[0]}}
    STEM_VERIFIER_MODEL=${STEM_VERIFIER_MODEL:-TIGER-Lab/general-verifier}
    STEM_VERIFIER_PORT=${STEM_VERIFIER_PORT:-18080}
    STEM_VERIFIER_GPU_MEMORY_UTILIZATION=${STEM_VERIFIER_GPU_MEMORY_UTILIZATION:-0.12}
    STEM_VERIFIER_MAX_MODEL_LEN=${STEM_VERIFIER_MAX_MODEL_LEN:-4096}
    STEM_VERIFIER_MAX_NUM_SEQS=${STEM_VERIFIER_MAX_NUM_SEQS:-16}
    STEM_VERIFIER_MAX_NEW_TOKENS=${STEM_VERIFIER_MAX_NEW_TOKENS:-256}
    STEM_VERIFIER_STARTUP_POLLS=${STEM_VERIFIER_STARTUP_POLLS:-120}
    STEM_VERIFIER_SLEEP_DURING_NON_REWARD=${STEM_VERIFIER_SLEEP_DURING_NON_REWARD:-1}
    STEM_VERIFIER_LIFECYCLE_TIMEOUT_SECONDS=${STEM_VERIFIER_LIFECYCLE_TIMEOUT_SECONDS:-120}

    export CODER1_EXEC=sandboxfusion
    export SANDBOX_FUSION_SERVERS=${SANDBOX_FUSION_SERVERS:-127.0.0.1}
    SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-http://127.0.0.1:8080/run_code}
    SANDBOX_FUSION_HEALTH_URL=${SANDBOX_FUSION_HEALTH_URL:-http://127.0.0.1:8080/v1/ping}

    STEM_VERIFIER_PID=""
}

check_sandbox() {
    if [[ "${NEED_SANDBOX}" -eq 1 ]]; then
        if [[ "$(curl -fsS --max-time 10 "${SANDBOX_FUSION_HEALTH_URL}")" != '"pong"' ]]; then
            echo "SandboxFusion health check failed" >&2
            exit 4
        fi
    fi
}

cleanup_local_stem_verifier() {
    if [[ -n "${STEM_VERIFIER_PID}" ]] && kill -0 "${STEM_VERIFIER_PID}" 2>/dev/null; then
        echo "Stopping local STEM verifier (pid=${STEM_VERIFIER_PID})"
        kill "${STEM_VERIFIER_PID}"
        wait "${STEM_VERIFIER_PID}" 2>/dev/null || true
    fi
}

start_local_stem_verifier() {
    if [[ ! -x "${VLLM_BIN}" ]]; then
        echo "vLLM executable is missing: ${VLLM_BIN}" >&2
        exit 5
    fi

    local verifier_url="http://127.0.0.1:${STEM_VERIFIER_PORT}"
    local verifier_dev_mode=0
    local -a verifier_sleep_args=()
    if [[ "${STEM_VERIFIER_SLEEP_DURING_NON_REWARD}" == "1" ]]; then
        verifier_dev_mode=1
        verifier_sleep_args+=(--enable-sleep-mode)
    fi

    echo "Starting local STEM verifier"
    echo "  model=${STEM_VERIFIER_MODEL}"
    echo "  shared_training_gpu=${STEM_VERIFIER_GPU}"
    echo "  gpu_memory_utilization=${STEM_VERIFIER_GPU_MEMORY_UTILIZATION}"
    echo "  sleep_during_non_reward=${STEM_VERIFIER_SLEEP_DURING_NON_REWARD}"
    echo "  log=${STEM_VERIFIER_LOG}"

    CUDA_VISIBLE_DEVICES="${STEM_VERIFIER_GPU}" \
        VLLM_USE_V1="${VLLM_USE_V1}" \
        VLLM_SERVER_DEV_MODE="${verifier_dev_mode}" \
        HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
        PYTHONNOUSERSITE=1 \
        "${VLLM_BIN}" serve "${STEM_VERIFIER_MODEL}" \
        --served-model-name TIGER-Lab/general-verifier \
        --host 127.0.0.1 \
        --port "${STEM_VERIFIER_PORT}" \
        --gpu-memory-utilization "${STEM_VERIFIER_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${STEM_VERIFIER_MAX_MODEL_LEN}" \
        --max-num-seqs "${STEM_VERIFIER_MAX_NUM_SEQS}" \
        --override-generation-config "{\"max_new_tokens\":${STEM_VERIFIER_MAX_NEW_TOKENS}}" \
        "${verifier_sleep_args[@]}" \
        >"${STEM_VERIFIER_LOG}" 2>&1 &
    STEM_VERIFIER_PID=$!

    local verifier_ready=0
    local attempt
    for ((attempt = 1; attempt <= STEM_VERIFIER_STARTUP_POLLS; attempt++)); do
        if ! kill -0 "${STEM_VERIFIER_PID}" 2>/dev/null; then
            echo "Local STEM verifier exited during startup; see ${STEM_VERIFIER_LOG}" >&2
            tail -n 100 "${STEM_VERIFIER_LOG}" >&2 || true
            exit 5
        fi
        if curl -fsS --max-time 2 "${verifier_url}/v1/models" >/dev/null; then
            verifier_ready=1
            break
        fi
        sleep 2
    done

    if [[ "${verifier_ready}" -ne 1 ]]; then
        echo "Local STEM verifier did not become ready; see ${STEM_VERIFIER_LOG}" >&2
        tail -n 100 "${STEM_VERIFIER_LOG}" >&2 || true
        exit 5
    fi

    export STEM_LLM_JUDGE_URL="${verifier_url}"
    if [[ "${STEM_VERIFIER_SLEEP_DURING_NON_REWARD}" == "1" ]]; then
        export STEM_VERIFIER_LIFECYCLE_CONTROL=1
        export STEM_VERIFIER_LIFECYCLE_URL="${verifier_url}"
        export STEM_VERIFIER_LIFECYCLE_TIMEOUT_SECONDS
        "${PYTHON_BIN}" -m verl.utils.reward_score.stem_llm_judge.lifecycle \
            sleep \
            --url "${verifier_url}" \
            --timeout "${STEM_VERIFIER_LIFECYCLE_TIMEOUT_SECONDS}"
        echo "Local STEM verifier sleep/wake lifecycle is enabled"
    else
        export STEM_VERIFIER_LIFECYCLE_CONTROL=0
        unset STEM_VERIFIER_LIFECYCLE_URL
    fi
    echo "Local STEM verifier is ready at ${STEM_LLM_JUDGE_URL}"
}

ensure_stem_verifier() {
    if [[ "${NEED_STEM_VERIFIER}" -eq 1 ]]; then
        if [[ -n "${STEM_LLM_JUDGE_URL:-}" ]]; then
            # Never send lifecycle commands to an externally managed verifier.
            export STEM_VERIFIER_LIFECYCLE_CONTROL=0
            unset STEM_VERIFIER_LIFECYCLE_URL
            if ! curl -fsS --max-time 5 "${STEM_LLM_JUDGE_URL%/}/v1/models" >/dev/null; then
                echo "External STEM verifier health check failed: ${STEM_LLM_JUDGE_URL}" >&2
                exit 5
            fi
            export STEM_LLM_JUDGE_URL
            echo "Using external STEM verifier at ${STEM_LLM_JUDGE_URL}"
        elif [[ "${AUTO_START_STEM_VERIFIER}" == "1" ]]; then
            start_local_stem_verifier
        else
            echo "Full-domain training requires STEM_LLM_JUDGE_URL or AUTO_START_STEM_VERIFIER=1" >&2
            exit 5
        fi
    fi
}
