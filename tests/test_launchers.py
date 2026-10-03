"""Configuration contracts for the Qwen3 and OLMo3 training recipes."""

import json
import os
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from launcher_support import ROOT, capture_launcher, training_contract


@pytest.mark.parametrize("model", ["qwen3_4b", "olmo3_7b_instruct_dpo"])
def test_training_contract_matches_recipe(model, tmp_path):
    expected = json.loads((ROOT / "tests/fixtures/launcher_contracts.json").read_text())[model]
    actual = capture_launcher(ROOT / f"scripts/train/poise_{model}.sh", tmp_path)
    assert training_contract(actual) == expected
    assert actual["argv"][:2] == ["-m", "poise.main"]
    assert actual["settings"]["BOOTSTRAP_STEPS"] == "16"


@pytest.mark.parametrize("model", ["qwen3_4b", "olmo3_7b_instruct_dpo"])
def test_launcher_composes_resolved_config_without_data_or_gpu(model, tmp_path):
    environment = os.environ.copy()
    environment.update(CONFIG_ONLY="1", PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE="1",
                       TRAIN_DATA_DIR=str(tmp_path / "missing-train"), VAL_DATA_DIR=str(tmp_path / "missing-val"),
                       OUTPUT_DIR=str(tmp_path / "output"), LOG_DIR=str(tmp_path / "logs"))
    result = subprocess.run(["bash", str(ROOT / f"scripts/train/poise_{model}.sh")],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
    config = OmegaConf.create(result.stdout[result.stdout.index("actor_rollout_ref:\n"):])
    assert config.trainer.poise.bootstrap.from_scratch is True
    assert config.trainer.poise.bootstrap.steps == 16
    assert "filter_groups" not in config.algorithm
    assert "group_filter" not in config.trainer.poise
    assert set(config.trainer.poise.bootstrap) == {"from_scratch", "steps", "rloo_batch_order"}
    assert config.actor_rollout_ref.rollout.n == 2
    assert config.data.train_batch_size == 256
    assert config.trainer.total_training_steps == 200
    assert config.algorithm.adv_estimator == "grpo"
    assert config.trainer.val_before_train is True
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "logs").exists()


def test_bootstrap_can_be_overridden(tmp_path):
    actual = capture_launcher(ROOT / "scripts/train/poise_qwen3_4b.sh", tmp_path,
                              {"POISE_BOOTSTRAP_STEPS": "4", "TOTAL_STEPS": "6"})
    assert actual["settings"]["BOOTSTRAP_STEPS"] == "4"
    assert "trainer.total_training_steps=6" in actual["argv"]


def test_estimator_validation_enables_unique_multi_rollout_protocol(tmp_path):
    actual = capture_launcher(ROOT / "scripts/train/poise_qwen3_4b.sh", tmp_path,
                              {"POISE_ESTIMATOR_EVAL_ON_VAL": "true"})
    assert actual["settings"]["VALIDATION_ROLLOUTS"] == "8"


def test_config_inspection_does_not_load_initial_estimators(tmp_path):
    capture_launcher(ROOT / "scripts/train/poise_qwen3_4b.sh", tmp_path,
                     {"POISE_MATH_ESTIMATOR_PATH": "/missing/math.joblib"})


def test_launcher_forwards_hydra_overrides(tmp_path):
    actual = capture_launcher(ROOT / "scripts/train/poise_qwen3_4b.sh", tmp_path,
                              extra_args=["trainer.save_freq=1", "trainer.resume_mode=disable"])
    assert actual["argv"][-2:] == ["trainer.save_freq=1", "trainer.resume_mode=disable"]


def test_pretrained_mode_defaults_to_zero_bootstrap_steps(tmp_path):
    actual = capture_launcher(ROOT / "scripts/train/poise_qwen3_4b.sh", tmp_path,
                              {"POISE_BOOTSTRAP_FROM_SCRATCH": "False"})
    assert actual["settings"]["BOOTSTRAP_STEPS"] == "0"
