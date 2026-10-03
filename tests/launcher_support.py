"""Inspect launcher arguments without starting a model, a service, or Ray."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def capture_launcher(launcher, tmp_path, overrides=None, extra_args=()):
    capture = tmp_path / "capture-python"
    capture.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "keys = ('BOOTSTRAP_FROM_SCRATCH', 'BOOTSTRAP_STEPS', "
        "'ESTIMATOR_WARMUP_STEPS', 'MATH_BUFFER_MAX_ROWS', 'CODE_BUFFER_MAX_ROWS', "
        "'OTHER_BUFFER_MAX_ROWS', 'ESTIMATOR_EVAL_ON_VAL', 'VALIDATION_ROLLOUTS')\n"
        "prefix = 'POISE_'\n"
        "payload = {'argv': sys.argv[1:], 'settings': {k: os.getenv(prefix+k) for k in keys}, "
        "'features': json.load(open(os.environ[prefix+'FEATURE_BUILDER_CONFIG'])), "
        "'fit': json.load(open(os.environ[prefix+'ESTIMATOR_FIT_CONFIG']))}\n"
        "print('CAPTURE_JSON=' + json.dumps(payload))\n"
    )
    capture.chmod(0o755)
    environment = launcher_environment(tmp_path, capture)
    environment.update(overrides or {})
    result = subprocess.run(["bash", str(launcher), *extra_args], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, check=True)
    line = next(line for line in result.stdout.splitlines() if line.startswith("CAPTURE_JSON="))
    return json.loads(line.removeprefix("CAPTURE_JSON="))


def launcher_environment(tmp_path, python_bin):
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("POISE_", "WANDB_"))
        and key not in {"MODEL_PATH", "EXPERIMENT_NAME", "TOTAL_STEPS", "TOTAL_EPOCHS", "TRAIN_SCOPE",
                       "TRAIN_PROMPT_BATCH_SIZE", "GEN_PROMPT_BATCH_SIZE", "VAL_BEFORE_TRAIN",
                       "ROLLOUT_GPU_MEMORY_UTILIZATION", "EXPERIMENT_SEED", "DATA_SHUFFLE",
                       "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "TRAIN_ENTRYPOINT", "TRAIN_CONFIG_NAME"}
    }
    environment.update(
        CONFIG_ONLY="1", SKIP_MODEL_PREFLIGHT="1", SKIP_GPU_PREFLIGHT="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHON_BIN=str(python_bin), CUDA_VISIBLE_DEVICES="0,1",
        WANDB_MODE="disabled", LOG_DIR=str(tmp_path / "logs"), OUTPUT_DIR=str(tmp_path / "output"),
        RAY_TMPDIR=str(tmp_path / "ray"),
    )
    return environment


def training_contract(payload):
    ignored = {"trainer.project_name", "trainer.experiment_name", "trainer.default_local_dir",
               "trainer.validation_data_dir"}
    arguments = {}
    for argument in payload["argv"]:
        if "=" not in argument or argument.startswith("--"):
            continue
        key, value = argument.split("=", 1)
        if key in ignored:
            continue
        if key in {"data.train_files", "data.val_files"}:
            value = [Path(path).name for path in ast.literal_eval(value)]
        arguments[key] = value
    settings = dict(payload["settings"])
    for key in ("BOOTSTRAP_FROM_SCRATCH", "ESTIMATOR_EVAL_ON_VAL"):
        settings[key] = settings[key].lower() == "true"
    return {"arguments": arguments, "settings": settings,
            "features": payload["features"], "fit": payload["fit"]}
