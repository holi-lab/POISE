#!/usr/bin/env bash
# Runtime defaults, output paths, and training command.

configure_environment() {
    export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/vendor${PYTHONPATH:+:${PYTHONPATH}}"

    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS=${NUM_GPUS:-${#GPU_IDS[@]}}

    DEFAULT_PYTHON_BIN=$(command -v python3 || command -v python)
    if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-base}" != "base" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        DEFAULT_PYTHON_BIN=${CONDA_PREFIX}/bin/python
    fi
    PYTHON_BIN=${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}
    VLLM_BIN=${VLLM_BIN:-$(dirname "${PYTHON_BIN}")/vllm}
    MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
    TRAIN_SCOPE=${TRAIN_SCOPE:-full}
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
    TOTAL_STEPS=${TOTAL_STEPS:-}
    SAVE_FREQ=${SAVE_FREQ:-40}
    TEST_FREQ=${TEST_FREQ:-40}
    VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
    DATA_SHUFFLE=${DATA_SHUFFLE:-True}
    VALIDATION_SHUFFLE=${VALIDATION_SHUFFLE:-True}
    MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-70000}
    ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}
    PROJECT_NAME=${PROJECT_NAME:-POISE}
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-poise-n_2-qwen3-4b}
    TRAIN_ENTRYPOINT=poise.main
    TRAIN_CONFIG_NAME=poise_fsdp
    TRAIN_ALGORITHM_LABEL=poise
    ADV_ESTIMATOR=grpo
    USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
    KL_REWARD_COEF=${KL_REWARD_COEF:-0.0}
    USE_KL_LOSS=${USE_KL_LOSS:-False}
    KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
    KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
    readonly EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
    readonly ROLLOUT_ENGINE_SEED=${ROLLOUT_ENGINE_SEED:-${EXPERIMENT_SEED}}

    readonly TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-256}
    readonly GEN_PROMPT_BATCH_SIZE=${GEN_PROMPT_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
    readonly PPO_PROMPT_MINI_BATCH_SIZE=${PPO_PROMPT_MINI_BATCH_SIZE:-32}
    readonly ROLLOUTS_PER_PROMPT=${ROLLOUTS_PER_PROMPT:-2}
    readonly MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
    readonly MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
    readonly MAX_MODEL_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    readonly VALIDATION_TEMPERATURE=0.6
    readonly VALIDATION_TOP_P=0.95
    readonly VALIDATION_TOP_K=-1
    readonly VALIDATION_SAMPLES_PER_ROW=1

    export PYTHONNOUSERSITE=1
    export PYTHONHASHSEED="${EXPERIMENT_SEED}"
    # FSDP Ray workers use this environment variable to seed Python, NumPy, and
    # Torch independently per data-parallel rank.
    export VERL_GLOBAL_SEED="${EXPERIMENT_SEED}"
    export HYDRA_FULL_ERROR=1
    export VLLM_USE_V1=${VLLM_USE_V1:-1}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export TOKENIZERS_PARALLELISM=true
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
    export WANDB_MODE=${WANDB_MODE:-disabled}
    # Never store credentials in this launcher.
    export WANDB_ENTITY=${WANDB_ENTITY:-}
    export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
    export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
    export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}
    export WANDB_CONSOLE=${WANDB_CONSOLE:-off}
    # Record the bias-corrected Adam moment-state variance proxy after each actor update.
    export VERL_ENABLE_ADAM_VARIANCE=${VERL_ENABLE_ADAM_VARIANCE:-1}
    unset RAY_ADDRESS

    POISE_OUTPUT_ROOT=${POISE_OUTPUT_ROOT:-${REPO_ROOT}/outputs}
    OUTPUT_DIR=${OUTPUT_DIR:-${POISE_OUTPUT_ROOT}/${EXPERIMENT_NAME}}
    CHECKPOINT_DIR=${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}
    VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${CHECKPOINT_DIR}/validation_jsonl}
    LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/training}
    if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
        LOG_FILE=${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}-config.log}
    else
        LOG_FILE=${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}
    fi
    STEM_VERIFIER_LOG=${STEM_VERIFIER_LOG:-${LOG_DIR}/${EXPERIMENT_NAME}-stem-verifier.log}
    export WANDB_DIR=${WANDB_DIR:-${OUTPUT_DIR}/wandb}
    # Keep the W&B identity across checkpoint resumes.
    export WANDB_RUN_ID_FILE=${WANDB_RUN_ID_FILE:-${OUTPUT_DIR}/wandb_run_id.txt}
    if [[ "${VERL_ENABLE_ADAM_VARIANCE}" == "1" ]]; then
        export VERL_ADAM_VARIANCE_LOG_PATH=${VERL_ADAM_VARIANCE_LOG_PATH:-${OUTPUT_DIR}/adam_variance_proxy.jsonl}
        export VERL_ADAM_VARIANCE_RUN_NAME=${VERL_ADAM_VARIANCE_RUN_NAME:-${EXPERIMENT_NAME}}
        export VERL_ADAM_VARIANCE_ALGORITHM=${VERL_ADAM_VARIANCE_ALGORITHM:-${TRAIN_ALGORITHM_LABEL}}
        export VERL_ADAM_VARIANCE_MODEL=${VERL_ADAM_VARIANCE_MODEL:-${MODEL_PATH}}
        export VERL_ADAM_VARIANCE_ROLLOUT_N=${VERL_ADAM_VARIANCE_ROLLOUT_N:-${ROLLOUTS_PER_PROMPT}}
        export VERL_ADAM_VARIANCE_TRAIN_PROMPT_BATCH_SIZE=${VERL_ADAM_VARIANCE_TRAIN_PROMPT_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
        export VERL_ADAM_VARIANCE_TRAIN_TRAJECTORIES=${VERL_ADAM_VARIANCE_TRAIN_TRAJECTORIES:-$((TRAIN_PROMPT_BATCH_SIZE * ROLLOUTS_PER_PROMPT))}
    fi
    RAY_GPU_TAG=${CUDA_VISIBLE_DEVICES//,/}
    export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/poise-ray-${RAY_GPU_TAG}-$$}
}

prepare_output_directories() {
    mkdir -p \
        "${OUTPUT_DIR}" \
        "${CHECKPOINT_DIR}" \
        "${VALIDATION_DATA_DIR}" \
        "${LOG_DIR}" \
        "${RAY_TMPDIR}" \
        "${WANDB_DIR}"
}

build_train_command() {
    local train_files val_files TRAINER_LOGGERS
    local -a TRAIN_LENGTH_ARGS HYDRA_ARGS

    train_files=$(printf "'%s'," "${TRAIN_FILES[@]}")
    train_files="[${train_files%,}]"
    val_files=$(printf "'%s'," "${VAL_FILES[@]}")
    val_files="[${val_files%,}]"

    if [[ "${WANDB_MODE}" == "disabled" ]]; then
        TRAINER_LOGGERS="['console']"
    else
        TRAINER_LOGGERS="['console','wandb']"
    fi

    TRAIN_LENGTH_ARGS=(
        trainer.total_epochs="${TOTAL_EPOCHS}"
    )
    if [[ -n "${TOTAL_STEPS}" ]]; then
        TRAIN_LENGTH_ARGS+=(trainer.total_training_steps="${TOTAL_STEPS}")
    fi

    HYDRA_ARGS=()
    if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
        HYDRA_ARGS+=(--cfg job --resolve)
    fi

    train_command=(
        "${PYTHON_BIN}" -m "${TRAIN_ENTRYPOINT}"
        "${HYDRA_ARGS[@]}"
        --config-path=config
        --config-name="${TRAIN_CONFIG_NAME}"

        # Algorithm
        algorithm.adv_estimator="${ADV_ESTIMATOR}"
        algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}"
        algorithm.kl_ctrl.kl_coef="${KL_REWARD_COEF}"

        # Data
        data.train_files="${train_files}"
        data.val_files="${val_files}"
        +data.seed="${EXPERIMENT_SEED}"
        data.shuffle="${DATA_SHUFFLE}"
        data.validation_shuffle="${VALIDATION_SHUFFLE}"
        data.prompt_key=prompt
        data.train_batch_size="${TRAIN_PROMPT_BATCH_SIZE}"
        data.gen_batch_size="${GEN_PROMPT_BATCH_SIZE}"
        data.max_prompt_length="${MAX_PROMPT_LENGTH}"
        data.max_response_length="${MAX_RESPONSE_LENGTH}"
        data.filter_overlong_prompts=True
        data.truncation=error

        # Actor
        actor_rollout_ref.model.path="${MODEL_PATH}"
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.model.enable_gradient_checkpointing=True
        actor_rollout_ref.actor.strategy=fsdp
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.optim.lr_warmup_steps=10
        actor_rollout_ref.actor.optim.weight_decay=0.01
        actor_rollout_ref.actor.optim.warmup_style=constant
        actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
        actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
        actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}"
        actor_rollout_ref.actor.clip_ratio_low=0.2
        actor_rollout_ref.actor.clip_ratio_high=0.28
        actor_rollout_ref.actor.clip_ratio_c=10.0
        actor_rollout_ref.actor.loss_agg_mode=token-mean
        actor_rollout_ref.actor.ppo_epochs=1
        actor_rollout_ref.actor.shuffle=False
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_PROMPT_MINI_BATCH_SIZE}"
        actor_rollout_ref.actor.ppo_micro_batch_size=null
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_MODEL_LENGTH * 2))
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        actor_rollout_ref.actor.entropy_coeff=0
        actor_rollout_ref.actor.grad_clip=1.0

        # Reference policy
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.ref.log_prob_micro_batch_size=null
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=null
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_MODEL_LENGTH * 2))

        # Rollout
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.mode=sync
        +actor_rollout_ref.rollout.engine_seed="${ROLLOUT_ENGINE_SEED}"
        actor_rollout_ref.rollout.n="${ROLLOUTS_PER_PROMPT}"
        actor_rollout_ref.rollout.temperature=1.0
        actor_rollout_ref.rollout.top_p=1.0
        actor_rollout_ref.rollout.top_k=-1
        actor_rollout_ref.rollout.val_kwargs.do_sample=True
        actor_rollout_ref.rollout.val_kwargs.temperature="${VALIDATION_TEMPERATURE}"
        actor_rollout_ref.rollout.val_kwargs.top_p="${VALIDATION_TOP_P}"
        actor_rollout_ref.rollout.val_kwargs.top_k="${VALIDATION_TOP_K}"
        actor_rollout_ref.rollout.val_kwargs.n="${VALIDATION_SAMPLES_PER_ROW}"
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_micro_batch_size=null
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=null
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_MODEL_LENGTH * 2))
        actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH}"
        actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_MODEL_LENGTH * 2))
        actor_rollout_ref.rollout.max_num_seqs=128
        actor_rollout_ref.rollout.enable_chunked_prefill=True
        actor_rollout_ref.rollout.multi_turn.enable=False

        # Rewards
        reward_model.reward_manager=async_multi_process_stem_lifecycle
        reward_model.overlong_buffer.enable=False
        reward_model.overlong_buffer.len=0
        reward_model.overlong_buffer.penalty_factor=0.0
        reward_model.overlong_buffer.log=False
        reward_model.sandbox_fusion.url="${SANDBOX_FUSION_URL}"
        reward_model.sandbox_fusion.max_concurrent=32
        reward_model.sandbox_fusion.memory_limit_mb=1024

        # Trainer
        trainer.logger="${TRAINER_LOGGERS}"
        trainer.project_name="${PROJECT_NAME}"
        trainer.experiment_name="${EXPERIMENT_NAME}"
        trainer.n_gpus_per_node="${NUM_GPUS}"
        trainer.nnodes=1
        trainer.balance_batch=True
        trainer.val_before_train="${VAL_BEFORE_TRAIN}"
        trainer.save_freq="${SAVE_FREQ}"
        trainer.test_freq="${TEST_FREQ}"
        "${TRAIN_LENGTH_ARGS[@]}"
        trainer.resume_mode=auto
        trainer.log_val_generations=20
        trainer.default_local_dir="${CHECKPOINT_DIR}"
        trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
        ray_init.num_cpus="${RAY_NUM_CPUS:-64}"
        "$@"
    )
}

print_launch_summary() {
    echo "Starting ${TRAIN_ALGORITHM_LABEL} training"
    echo "  entrypoint=${TRAIN_ENTRYPOINT} config=${TRAIN_CONFIG_NAME} adv_estimator=${ADV_ESTIMATOR}"
    echo "  experiment=${EXPERIMENT_NAME}"
    echo "  output=${OUTPUT_DIR}"
    echo "  model=${MODEL_PATH}"
    echo "  scope=${TRAIN_SCOPE} train_files=${#TRAIN_FILES[@]} val_files=${#VAL_FILES[@]}"
    echo "  experiment_seed=${EXPERIMENT_SEED}"
    echo "  rollout_engine_seed=${ROLLOUT_ENGINE_SEED}"
    echo "  train_shuffle=${DATA_SHUFFLE}"
    echo "  GPUs=${CUDA_VISIBLE_DEVICES}"
    echo "  prompt_batch=${TRAIN_PROMPT_BATCH_SIZE} gen_prompt_batch=${GEN_PROMPT_BATCH_SIZE} prompt_minibatch=${PPO_PROMPT_MINI_BATCH_SIZE}"
    echo "  rollout_n=${ROLLOUTS_PER_PROMPT} response_length=${MAX_RESPONSE_LENGTH}"
    echo "  rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    echo "  kl_in_reward=${USE_KL_IN_REWARD} kl_reward_coef=${KL_REWARD_COEF}"
    echo "  kl_loss=${USE_KL_LOSS} kl_loss_coef=${KL_LOSS_COEF} kl_loss_type=${KL_LOSS_TYPE}"
    echo "  validation_sampling=true temperature=${VALIDATION_TEMPERATURE} top_p=${VALIDATION_TOP_P} n=${VALIDATION_SAMPLES_PER_ROW}"
    echo "  wandb_mode=${WANDB_MODE} entity=${WANDB_ENTITY} project=${PROJECT_NAME}"
    if [[ "${WANDB_MODE}" != "disabled" ]]; then
        echo "  wandb_run_id_file=${WANDB_RUN_ID_FILE}"
    fi
    echo "  checkpoints=${CHECKPOINT_DIR}"
    echo "  validation_jsonl=${VALIDATION_DATA_DIR}"
    if [[ "${VERL_ENABLE_ADAM_VARIANCE}" == "1" ]]; then
        echo "  adam_variance_jsonl=${VERL_ADAM_VARIANCE_LOG_PATH}"
    fi
    if [[ "${NEED_STEM_VERIFIER}" -eq 1 && -n "${STEM_LLM_JUDGE_URL:-}" ]]; then
        echo "  stem_verifier=${STEM_LLM_JUDGE_URL}"
    fi
}
