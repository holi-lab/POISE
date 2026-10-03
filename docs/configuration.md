# Configuration

POISE defaults: [`_poise.sh`](../scripts/train/_poise.sh) and [`poise_fsdp.yaml`](../src/poise/config/poise_fsdp.yaml). [`_launch.sh`](../scripts/train/_launch.sh) coordinates helpers in [`scripts/train/lib/`](../scripts/train/lib/):

| File | Responsibility |
| --- | --- |
| `config.sh` | Runtime defaults, output paths, and training command |
| `data.sh` | Dataset splits |
| `checks.sh` / `check_model.py` | Input, GPU, and model checks |
| `services.sh` | SandboxFusion and STEM verifier lifecycle |

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Training GPUs |
| `EXPERIMENT_NAME` | Model-specific | Output directory and resume identity |
| `EXPERIMENT_SEED` | `42` | Training and rollout seed |
| `TOTAL_STEPS` | `200` | Actor updates, including bootstrap |
| `POISE_BOOTSTRAP_STEPS` | `16` | RLOO bootstrap updates |
| `TRAIN_PROMPT_BATCH_SIZE` | `256` | Prompts per update |
| `TRAIN_DATA_DIR` / `VAL_DATA_DIR` | `data/train` / `data/online_eval` | Dataset locations |
| `WANDB_MODE` | `disabled` | Set `online` after `wandb login` to enable W&B |

Environment variables and trailing Hydra arguments override defaults:

```bash
EXPERIMENT_NAME=poise-qwen3-seed7 EXPERIMENT_SEED=7 TOTAL_STEPS=400 \
  bash scripts/train/poise_qwen3_4b.sh trainer.save_freq=20
```

Scratch bootstrap fits all three domain probes; use the default `TRAIN_SCOPE=full`. To start from fitted probes, set `POISE_BOOTSTRAP_FROM_SCRATCH=False` and provide `POISE_MATH_ESTIMATOR_PATH`, `POISE_CODE_ESTIMATOR_PATH`, and `POISE_OTHER_ESTIMATOR_PATH` matching the feature/fit configs. Bootstrap steps then default to zero.

## Reward services

For a remote SandboxFusion service, configure `SANDBOX_FUSION_SERVERS` (comma-separated hosts), `SANDBOX_FUSION_URL` (the `/run_code` endpoint), and `SANDBOX_FUSION_HEALTH_URL` (the `/v1/ping` endpoint). `STEM_LLM_JUDGE_URL` selects an existing STEM verifier; otherwise the launcher starts `TIGER-Lab/general-verifier` on the first training GPU.

## Checkpoints

Outputs default to `outputs/<experiment>/`; override `OUTPUT_DIR` to relocate them. Repeat the same command to resume, keeping bootstrap steps and probe settings unchanged. Use a new `EXPERIMENT_NAME` to start a separate run.

Checkpoints include the actor, probes, buffers, and training phase. Probe snapshots are saved every 10 steps (`POISE_ESTIMATOR_SAVE_FREQ`). Per-prompt rewards and predictions are in `checkpoints/prompt_reward_logs/`.

## Probe validation

Set `POISE_ESTIMATOR_EVAL_ON_VAL=True` to score probes with eight rollouts per unique validation prompt. First prepare unique AIME and AMC files:

```bash
python -m poise.prepare_unique_validation \
  --input data/online_eval/math__aime_repeated_8x_240.parquet \
  --output data/online_eval/poise/math__aime_unique_30.parquet --expected-rows 30
python -m poise.prepare_unique_validation \
  --input data/online_eval/math__amc_repeated_4x_332.parquet \
  --output data/online_eval/poise/math__amc_unique_83.parquet --expected-rows 83
```

Initial validation is enabled by default; set `VAL_BEFORE_TRAIN=False` to skip it. Combining probe validation with initial validation requires fitted initial probes.

## Tests

```bash
python -m pip install -e '.[test]'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests -q
```
