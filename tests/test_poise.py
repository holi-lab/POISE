import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from poise.trainer import RayPOISETrainer
from poise.core import (
    compute_group_targets_and_cross_baselines,
    regression_diagnostic_values,
)
from poise.domain_estimator_bank import (
    DOMAIN_NAMES,
    DomainEstimatorBank,
    DomainEstimatorState,
    RecentDomainBuffer,
    domain_from_data_source,
)
from poise.estimator import (
    EstimatorConfig,
    EstimatorFitConfig,
    EstimatorModelConfig,
    FastPCA,
    FeatureBuilderConfig,
    MultiEstimatorRegressor,
    ProjectionConfig,
    SingleTrajectoryEstimator,
    SingleTrajectoryFeatureBuilder,
    fit_estimator,
)
from poise.main import (
    apply_poise_validation_file_config,
    apply_poise_validation_rollout_config,
    seed_poise_task_runner,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


def test_ray_task_receives_config_without_driver_local_resolvers(monkeypatch):
    from poise import main as entrypoint

    config = OmegaConf.create({
        "trainer": {"profile_steps": None, "poise": {"estimator": {
            "eval_on_val": False,
            "feature_builder_config_path": "${poise_config:feature_builder_config_qwen3_4b.json}",
        }}},
    })
    received = []

    def remote_run(worker_config):
        # Serialized values must be usable by a worker that has never
        # registered the driver's custom resolver.
        payload = OmegaConf.to_container(worker_config, resolve=False)
        path = payload["trainer"]["poise"]["estimator"]["feature_builder_config_path"]
        assert "${" not in path
        assert Path(path).is_file()
        received.append(path)
        return None

    runner = SimpleNamespace(run=SimpleNamespace(remote=remote_run))
    monkeypatch.setattr(entrypoint, "TaskRunner", SimpleNamespace(remote=lambda: runner))
    monkeypatch.setattr(entrypoint.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(entrypoint.ray, "get", lambda value: value)
    entrypoint.run_ppo(config)
    assert len(received) == 1


def test_poise_validation_rollouts_override_shared_launcher_value_when_enabled():
    config = OmegaConf.create(
        {
            "trainer": {
                "poise": {
                    "validation_rollouts_per_prompt": 8,
                    "estimator": {"eval_on_val": True},
                }
            },
            "actor_rollout_ref": {
                "rollout": {"val_kwargs": {"n": 1}}
            },
        }
    )

    apply_poise_validation_rollout_config(config)

    assert config.actor_rollout_ref.rollout.val_kwargs.n == 8


def test_poise_prompt_normalizes_nested_non_tensor_batch_rows():
    homogeneous = DataProto.from_single_dict(
        {
            "input_ids": torch.tensor([[1], [2]]),
            "solutions": np.asarray([["first"], ["second"]], dtype=object),
        }
    )
    heterogeneous = DataProto.from_single_dict(
        {
            "input_ids": torch.tensor([[3], [4]]),
            "solutions": np.asarray(
                [
                    ["third"],
                    ["fourth", "alternate"],
                ],
                dtype=object,
            ),
        }
    )

    homogeneous = RayPOISETrainer._normalize_non_tensor_rows(
        homogeneous
    )
    heterogeneous = RayPOISETrainer._normalize_non_tensor_rows(
        heterogeneous
    )
    combined = DataProto.concat([homogeneous, heterogeneous])

    assert homogeneous.non_tensor_batch["solutions"].shape == (2,)
    assert heterogeneous.non_tensor_batch["solutions"].shape == (2,)
    assert combined.non_tensor_batch["solutions"].tolist() == [
        ["first"],
        ["second"],
        ["third"],
        ["fourth", "alternate"],
    ]


def test_poise_validation_rollouts_preserve_shared_launcher_value_when_disabled():
    config = OmegaConf.create(
        {
            "trainer": {
                "poise": {
                    "validation_rollouts_per_prompt": 8,
                    "estimator": {"eval_on_val": False},
                }
            },
            "actor_rollout_ref": {
                "rollout": {"val_kwargs": {"n": 1}}
            },
        }
    )

    apply_poise_validation_rollout_config(config)

    assert config.actor_rollout_ref.rollout.val_kwargs.n == 1


def test_poise_config_accepts_rloo_validation_sampling_when_estimator_eval_disabled():
    trainer = object.__new__(RayPOISETrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                n=2,
                val_kwargs=SimpleNamespace(n=1),
            ),
            actor=SimpleNamespace(strategy="fsdp"),
        ),
        algorithm=SimpleNamespace(adv_estimator="grpo"),
        trainer=SimpleNamespace(
            poise=SimpleNamespace(
                validation_rollouts_per_prompt=8,
                bootstrap=SimpleNamespace(
                    from_scratch=True,
                    steps=8,
                ),
                estimator=OmegaConf.create(
                    {
                        "group_size": 2,
                        "eval_on_val": False,
                        "save_freq": 10,
                    }
                ),
            )
        ),
    )

    trainer._validate_poise_config()


def test_qwen_poise_replaces_only_repeated_validation_files(
    monkeypatch,
    tmp_path,
):
    aime_unique = tmp_path / "math__aime_unique_30.parquet"
    amc_unique = tmp_path / "math__amc_unique_83.parquet"
    aime_unique.touch()
    amc_unique.touch()
    monkeypatch.setenv("POISE_AIME_VAL_FILE", str(aime_unique))
    monkeypatch.setenv("POISE_AMC_VAL_FILE", str(amc_unique))
    untouched = "/validation/math__math_500.parquet"
    config = OmegaConf.create(
        {
            "trainer": {
                "poise": {"estimator": {"eval_on_val": True}}
            },
            "data": {
                "val_files": [
                    "/validation/math__aime_repeated_8x_240.parquet",
                    "/validation/math__amc_repeated_4x_332.parquet",
                    untouched,
                ]
            }
        }
    )

    apply_poise_validation_file_config(config)

    assert list(config.data.val_files) == [
        str(aime_unique.resolve()),
        str(amc_unique.resolve()),
        untouched,
    ]


def test_poise_validation_files_preserve_shared_launcher_value_when_disabled(
    monkeypatch,
    tmp_path,
):
    aime_unique = tmp_path / "math__aime_unique_30.parquet"
    amc_unique = tmp_path / "math__amc_unique_83.parquet"
    aime_unique.touch()
    amc_unique.touch()
    monkeypatch.setenv("POISE_AIME_VAL_FILE", str(aime_unique))
    monkeypatch.setenv("POISE_AMC_VAL_FILE", str(amc_unique))
    original_files = [
        "/validation/math__aime_repeated_8x_240.parquet",
        "/validation/math__amc_repeated_4x_332.parquet",
    ]
    config = OmegaConf.create(
        {
            "trainer": {
                "poise": {"estimator": {"eval_on_val": False}}
            },
            "data": {"val_files": original_files},
        }
    )

    apply_poise_validation_file_config(config)

    assert list(config.data.val_files) == original_files


def test_poise_validation_files_are_unchanged_without_opt_in(monkeypatch):
    monkeypatch.delenv("POISE_AIME_VAL_FILE", raising=False)
    monkeypatch.delenv("POISE_AMC_VAL_FILE", raising=False)
    original_files = [
        "/validation/math__aime_repeated_8x_240.parquet",
        "/validation/math__amc_repeated_4x_332.parquet",
    ]
    config = OmegaConf.create(
        {
            "trainer": {
                "poise": {"estimator": {"eval_on_val": True}}
            },
            "data": {"val_files": original_files},
        }
    )

    apply_poise_validation_file_config(config)

    assert list(config.data.val_files) == original_files


def test_poise_validation_rejects_zero_rollouts():
    config = OmegaConf.create(
        {
            "trainer": {
                "poise": {
                    "validation_rollouts_per_prompt": 0,
                    "estimator": {"eval_on_val": True},
                }
            },
            "actor_rollout_ref": {
                "rollout": {"val_kwargs": {"n": 1}}
            },
        }
    )

    with pytest.raises(ValueError, match="at least one rollout"):
        apply_poise_validation_rollout_config(config)


def test_poise_task_runner_seed_is_reproducible():
    config = OmegaConf.create({"data": {"seed": 43}})
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        seed_poise_task_runner(config)
        first = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        seed_poise_task_runner(config)
        second = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    assert first == second


@pytest.mark.parametrize("preserve_order", [False, True])
def test_actor_batch_indices_preserve_every_rollout(preserve_order):
    batch = DataProto.from_single_dict(
        {
            "responses": torch.zeros((6, 1), dtype=torch.long),
            "uid": np.asarray(["first", "second", "third", "third", "second", "first"]),
        }
    )

    selected = RayPOISETrainer._actor_batch_indices(
        batch,
        prompt_count=3,
        group_size=2,
        preserve_order=preserve_order,
    )

    assert selected == ([0, 1, 2, 3, 4, 5] if preserve_order else [0, 5, 1, 4, 2, 3])


def test_poise_missing_hidden_capture_reports_tensor_keys():
    batch = DataProto.from_single_dict(
        {
            "responses": torch.zeros((1, 1), dtype=torch.long),
            "response_mask": torch.ones((1, 1), dtype=torch.long),
        }
    )
    trainer = object.__new__(RayPOISETrainer)

    with pytest.raises(KeyError, match="estimator_prompt_hidden"):
        trainer._compute_poise_advantages(
            batch=batch,
            reward_sums=torch.zeros(1),
            entropies=torch.zeros((1, 1)),
        )


def test_n2_cross_rollout_is_sibling_swap_with_reordered_rows():
    rewards = torch.tensor([1.0, 0.0, 0.25, 0.75])
    predictions = torch.tensor([0.1, 0.2, 0.9, 0.8])
    groups = {"first": [0, 2], "second": [1, 3]}

    targets, baselines = compute_group_targets_and_cross_baselines(
        reward_sums=rewards,
        value_predictions=predictions,
        uid_to_indices=groups,
        group_size=2,
        target_mode="other_rollout_correctness",
    )

    torch.testing.assert_close(targets, torch.tensor([0.25, 0.75, 1.0, 0.0]))
    torch.testing.assert_close(baselines, torch.tensor([0.9, 0.8, 0.1, 0.2]))


def test_n3_cross_rollout_uses_leave_one_out_means():
    rewards = torch.tensor([0.0, 0.5, 1.0])
    predictions = torch.tensor([0.2, 0.4, 0.9])

    targets, baselines = compute_group_targets_and_cross_baselines(
        reward_sums=rewards,
        value_predictions=predictions,
        uid_to_indices={"prompt": [0, 1, 2]},
        group_size=3,
        target_mode="other_rollout_correctness",
    )

    torch.testing.assert_close(targets, torch.tensor([0.75, 0.5, 0.25]))
    torch.testing.assert_close(baselines, torch.tensor([0.65, 0.55, 0.3]))


def test_cross_rollout_rejects_incomplete_groups():
    with pytest.raises(ValueError, match="expected group_size=2"):
        compute_group_targets_and_cross_baselines(
            reward_sums=torch.tensor([1.0]),
            value_predictions=torch.tensor([0.4]),
            uid_to_indices={"prompt": [0]},
            group_size=2,
            target_mode="other_rollout_correctness",
        )


def test_regression_diagnostics_include_error_and_correlation_metrics():
    metrics = regression_diagnostic_values(
        predictions=[0.1, 0.4, 0.9],
        targets=[0.0, 0.5, 1.0],
    )

    assert metrics["target_mae"] == pytest.approx(0.1)
    assert metrics["target_rmse"] == pytest.approx(0.1)
    assert metrics["constant_brier"] == pytest.approx(1.0 / 6.0)
    assert metrics["brier_skill"] == pytest.approx(0.94)
    assert metrics["brier_skill_defined"] == 1.0
    assert metrics["target_bias"] == pytest.approx(-1.0 / 30.0)
    assert metrics["target_pearson"] > 0.98
    assert metrics["prediction_p50"] == pytest.approx(0.4)


def test_brier_skill_marks_constant_target_batches_as_undefined():
    metrics = regression_diagnostic_values(
        predictions=[0.2, 0.4],
        targets=[0.0, 0.0],
    )

    assert metrics["target_rmse"] ** 2 == pytest.approx(0.1)
    assert metrics["constant_brier"] == 0.0
    assert metrics["brier_skill"] == 0.0
    assert metrics["brier_skill_defined"] == 0.0


@pytest.mark.parametrize(
    ("data_source", "expected"),
    [
        ("math__combined", "math"),
        ("codegen__taco", "code"),
        ("logic__arcagi1", "other"),
        ("simulation__codeio", "other"),
        ("stem__web", "other"),
        ("table__hitab", "other"),
    ],
)
def test_domain_routing(data_source, expected):
    assert domain_from_data_source(data_source) == expected


def test_validation_estimator_metrics_use_response_loo_and_prompt_means():
    rows = []
    prompt_specs = [
        (
            "math",
            0,
            [1.0, 0.0, 1.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
        ),
        ("code", 1, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ]
    for domain, prompt_id, rewards, predictions in prompt_specs:
        reward_sum = sum(rewards)
        for index, (reward, prediction) in enumerate(
            zip(rewards, predictions, strict=True)
        ):
            rows.append(
                {
                    "domain": domain,
                    "prompt_id": prompt_id,
                    "prediction": prediction,
                    "reward": reward,
                    "loo_target": (reward_sum - rewards[index]) / 3.0,
                }
            )

    metrics = RayPOISETrainer._estimator_validation_metrics(rows)

    assert metrics[
        "val/poise_estimator/all/response_loo/target_rmse"
    ] == pytest.approx(0.0)
    assert metrics[
        "val/poise_estimator/all/prompt_mean/target_rmse"
    ] == pytest.approx(0.0)
    assert metrics["val/poise_estimator/all/num_responses"] == 8.0
    assert metrics["val/poise_estimator/all/num_prompts"] == 2.0
    assert metrics["val/poise_estimator/math/num_prompts"] == 1.0
    assert metrics["val/poise_estimator/code/num_prompts"] == 1.0


def test_compact_validation_metrics_keeps_one_canonical_variable():
    metrics = RayPOISETrainer._compact_validation_metrics(
        {
            "val-core/math/acc/mean@8": 0.5,
            "val-aux/math/reward/mean@8": 0.5,
            "val-aux/math/score/mean@8": 0.5,
            "val-core/code/reward/mean@8": 0.25,
            "val-aux/code/score/mean@8": 0.25,
            "val-aux/math/format_reward/mean@8": 1.0,
            "val/test_score/math/dataset": 0.5,
            "val/macro_mean": 0.375,
        }
    )

    assert "val-core/math/acc/mean@8" in metrics
    assert "val-aux/math/reward/mean@8" not in metrics
    assert "val-aux/math/score/mean@8" not in metrics
    assert "val-core/code/reward/mean@8" in metrics
    assert "val-aux/code/score/mean@8" not in metrics
    assert "val-aux/math/format_reward/mean@8" in metrics
    assert metrics["val/test_score/math/dataset"] == 0.5
    assert metrics["val/macro_mean"] == 0.375


def test_compact_wandb_metrics_keeps_canonical_poise_series():
    metrics = RayPOISETrainer._compact_wandb_metrics(
        {
            "poise/online/math/target_rmse": 0.2,
            "poise/online/math/online_target_rmse": 0.2,
            "poise/online/math/prediction_mean": 0.4,
            "poise/online/math/target_mean": 0.5,
            "poise/online/math/target_std": 0.5,
            "poise/online/math/target_bias": -0.1,
            "poise/online/math/target_p10": 0.0,
            "poise/online/math/prediction_p90": 0.8,
            "poise/online/math/advantage_mean": 0.1,
            "poise/online/math/advantage_std": 0.3,
            "poise/online/math/cross_rollout/group_size": 2.0,
            "poise/online/math/cross_rollout/value_prediction/mean": 0.4,
            "poise/online/math/reward_mean": 0.5,
            "poise/online/math/reward_std": 0.5,
            "poise/online/math/cross_rollout/reward/mean": 0.5,
            "poise/online/math/cross_rollout/reward/p10": 0.0,
            "poise/online/math/cross_rollout/cross_baseline/mean": 0.4,
            "poise/online/math/cross_rollout/advantage/p10": -0.4,
            "poise/online/math/pred_clip_min_count": 4.0,
            "poise/online/math/pred_clip_frac_min": 0.25,
            "poise/online/math/rows": 16.0,
            "poise/online/math/member_count": 0.0,
            "poise/estimator/math/rows_added": 16.0,
            "poise/estimator/math/buffer_rows": 96.0,
            "poise/estimator/math/recent_buffer_rows": 96.0,
            "poise/estimator/math/fit_rows": 96.0,
            "poise/estimator/math/rows": 96.0,
            "poise/estimator/math/buffer_max_rows": 96.0,
            "poise/estimator/math/observed_steps": 17.0,
            "poise/estimator/math/buffer_steps": 12.0,
            "poise/estimator/math/prediction_p10": 0.2,
            "poise/estimator/math/target_p90": 1.0,
            "poise/estimator/math/train_rmse": 0.1,
            "poise/prompt_reward_log/prompt_count": 256.0,
        }
    )

    assert "poise/online/math/target_rmse" in metrics
    assert "poise/online/math/online_target_rmse" not in metrics
    assert "poise/online/math/prediction_mean" in metrics
    assert (
        "poise/online/math/cross_rollout/value_prediction/mean"
        not in metrics
    )
    assert "poise/online/math/reward_mean" not in metrics
    assert "poise/online/math/reward_std" not in metrics
    assert "poise/online/math/cross_rollout/reward/mean" not in metrics
    assert "poise/online/math/cross_rollout/reward/p10" not in metrics
    assert (
        "poise/online/math/cross_rollout/cross_baseline/mean" not in metrics
    )
    assert "poise/online/math/cross_rollout/advantage/p10" not in metrics
    assert "poise/online/math/target_p10" not in metrics
    assert "poise/online/math/prediction_p90" not in metrics
    assert "poise/online/math/advantage_mean" not in metrics
    assert metrics["poise/online/math/advantage_std"] == 0.3
    assert "poise/online/math/pred_clip_min_count" not in metrics
    assert "poise/online/math/pred_clip_frac_min" in metrics
    assert "poise/online/math/member_count" not in metrics

    assert "poise/estimator/math/rows_added" not in metrics
    assert "poise/estimator/math/buffer_rows" not in metrics
    assert metrics["poise/estimator/math/recent_buffer_rows"] == 96.0
    assert metrics["poise/estimator/math/fit_rows"] == 96.0
    assert "poise/estimator/math/rows" not in metrics
    assert "poise/estimator/math/buffer_max_rows" not in metrics
    assert "poise/estimator/math/observed_steps" not in metrics
    assert "poise/estimator/math/buffer_steps" not in metrics
    assert "poise/estimator/math/prediction_p10" not in metrics
    assert "poise/estimator/math/target_p90" not in metrics
    assert metrics["poise/estimator/math/train_rmse"] == 0.1

    assert set(metrics).isdisjoint(
        {
            "poise/prompt_reward_log/prompt_count",
        }
    )


def test_compact_wandb_metrics_only_filters_poise_namespace():
    metrics = RayPOISETrainer._compact_wandb_metrics(
        {
            "critic/score/mean": 0.5,
            "critic/rewards/mean": 0.5,
            "critic/advantages/mean": 0.1,
            "critic/returns/mean": 0.1,
            "training/global_step": 17,
            "training/epoch": 1,
            "training/data_epoch": 1.25,
            "val/poise_estimator/all/num_responses": 64.0,
            "val/poise_estimator/all/response_loo/rows": 64.0,
            "poise/final_batch/math_prompts": 128.0,
        }
    )

    assert metrics["critic/rewards/mean"] == 0.5
    assert metrics["critic/returns/mean"] == 0.1
    assert metrics["training/global_step"] == 17
    assert metrics["training/epoch"] == 1
    assert metrics["training/data_epoch"] == 1.25
    assert metrics["val/poise_estimator/all/response_loo/rows"] == 64.0
    assert metrics["poise/final_batch/math_prompts"] == 128.0


def test_compact_wandb_metrics_preserves_non_pairwise_cross_rollout_metrics():
    metrics = RayPOISETrainer._compact_wandb_metrics(
        {
            "poise/online/math/cross_rollout/group_size": 4.0,
            "poise/online/math/cross_rollout/cross_baseline/p50": 0.4,
            "poise/online/math/cross_rollout/baseline_vs_reward/rmse": 0.2,
        }
    )

    assert metrics["poise/online/math/cross_rollout/group_size"] == 4.0
    assert (
        metrics["poise/online/math/cross_rollout/cross_baseline/p50"]
        == 0.4
    )


def test_scratch_initial_estimator_validation_restores_bootstrap_runtime():
    trainer = object.__new__(RayPOISETrainer)
    bootstrap_runtime = object()
    initial_bank = object()
    restored_runtimes = []
    trainer._poise_runtime = bootstrap_runtime
    trainer._initialize_estimator_bank = lambda: initial_bank
    trainer._set_estimator_runtime = restored_runtimes.append

    def fake_validate():
        assert trainer._poise_bank is initial_bank
        assert not hasattr(trainer, "_poise_validation_metric_root")
        return {
            "val/base": 0.5,
            "val/poise_estimator/all/num_responses": 8.0,
        }

    trainer._validate = fake_validate

    metrics = trainer._validate_initial_estimator_for_scratch_bootstrap()

    assert metrics == {
        "val/base": 0.5,
        "val/poise_estimator/all/num_responses": 8.0,
    }
    assert not hasattr(trainer, "_poise_bank")
    assert restored_runtimes == [bootstrap_runtime]


def test_validation_estimator_scores_existing_generation_without_regeneration():
    rollout_count = 4
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[10, 11]] * (rollout_count * 2)),
            "attention_mask": torch.ones((rollout_count * 2, 4)),
        },
        non_tensors={
            "data_source": np.asarray(
                ["math__math500"] * rollout_count
                + ["codegen__humaneval"] * rollout_count,
                dtype=object,
            )
        },
    )
    reward_tensor = torch.tensor(
        [[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [1.0], [1.0]]
    )
    predictions = torch.tensor(
        [1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0]
    )

    class ActorRolloutGroup:
        calls = 0

        def compute_log_prob(self, received_batch):
            self.calls += 1
            assert received_batch is batch
            assert "estimator_hidden_capture" in received_batch.meta_info
            assert "response_mask" in received_batch.batch
            return DataProto.from_dict(
                tensors={
                    "old_log_probs": torch.zeros((rollout_count * 2, 2)),
                    "entropys": torch.zeros((rollout_count * 2, 2)),
                    "estimator_prompt_hidden": torch.zeros(
                        (rollout_count * 2, 1)
                    ),
                    "estimator_response_hidden": torch.zeros(
                        (rollout_count * 2, 1)
                    ),
                }
            )

    trainer = object.__new__(RayPOISETrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            poise=SimpleNamespace(
                validation_rollouts_per_prompt=rollout_count
            )
        )
    )
    trainer._poise_capture_spec = {"layer_index": 1}
    trainer.actor_rollout_wg = ActorRolloutGroup()
    trainer._predict_poise_values = lambda **kwargs: (
        predictions,
        [None] * (rollout_count * 2),
        ["math"] * rollout_count + ["code"] * rollout_count,
        [{} for _ in range(rollout_count * 2)],
    )

    rows = trainer._estimator_validation_batch_rows(
        batch=batch,
        reward_tensor=reward_tensor,
        prompt_offset=5,
    )

    assert trainer.actor_rollout_wg.calls == 1
    assert "response_mask" not in batch.batch
    assert "estimator_hidden_capture" not in batch.meta_info
    assert [row["prompt_id"] for row in rows] == [5] * 4 + [6] * 4
    assert [row["loo_target"] for row in rows[:4]] == pytest.approx(
        [1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]
    )


def test_poise_validate_reuses_base_validation_reward_batch(monkeypatch):
    base_batch = object()
    reward_tensor = torch.ones((8, 1))
    reward_calls = []
    diagnostic_calls = []

    def val_reward_fn(batch, return_dict=False):
        reward_calls.append((batch, return_dict))
        return {"reward_tensor": reward_tensor}

    def fake_base_validate(self):
        assert self._poise_validation_dump_rows == []
        result = self.val_reward_fn(base_batch, return_dict=True)
        assert result["reward_tensor"] is reward_tensor
        assert len(self._poise_validation_dump_rows) == 8
        return {"val/base": 1.0}

    monkeypatch.setattr(RayPPOTrainer, "_validate", fake_base_validate)

    trainer = object.__new__(RayPOISETrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            poise=SimpleNamespace(
                validation_rollouts_per_prompt=8,
                estimator=SimpleNamespace(eval_on_val=True),
            )
        )
    )
    trainer._poise_bank = object()
    trainer.val_reward_fn = val_reward_fn
    trainer._estimator_validation_batch_rows = lambda **kwargs: (
        diagnostic_calls.append(kwargs)
        or [
            {
                "domain": "math",
                "prompt_id": 0,
                "prediction": 1.0,
                "reward": 1.0,
                "loo_target": 1.0,
            }
            for _ in range(8)
        ]
    )

    metrics = trainer._validate()

    assert reward_calls == [(base_batch, True)]
    assert len(diagnostic_calls) == 1
    assert diagnostic_calls[0]["batch"] is base_batch
    assert diagnostic_calls[0]["reward_tensor"] is reward_tensor
    assert trainer.val_reward_fn is val_reward_fn
    assert not hasattr(trainer, "_poise_validation_dump_rows")
    assert metrics["val/base"] == 1.0
    assert metrics["val/poise_estimator/all/num_responses"] == 8.0


def test_poise_validation_dump_records_raw_estimator_values(tmp_path):
    trainer = object.__new__(RayPOISETrainer)
    trainer.global_steps = 40
    trainer._poise_validation_dump_rows = [
        {
            "domain": "math",
            "prompt_id": 0,
            "prediction": 0.25,
            "reward": 0.0,
            "loo_target": 1.0,
        },
        {
            "domain": "math",
            "prompt_id": 0,
            "prediction": 0.75,
            "reward": 1.0,
            "loo_target": 0.0,
            "member_predictions": [0.7, 0.8],
        },
    ]

    trainer._dump_generations(
        inputs=["prompt", "prompt"],
        outputs=["failure", "success"],
        scores=[0.0, 1.0],
        reward_extra_infos_dict={"reward": [0.0, 1.0]},
        dump_path=str(tmp_path),
        metadata={
            "data_source": ["math__test", "math__test"],
            "dataset": ["test", "test"],
        },
        record_type="validation",
    )

    with (tmp_path / "40.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows[0]["prediction"] is None
    assert rows[0]["estimator_prediction"] == pytest.approx(0.25)
    assert rows[0]["estimator_loo_target"] == pytest.approx(1.0)
    assert rows[0]["estimator_error"] == pytest.approx(-0.75)
    assert rows[0]["estimator_domain"] == "math"
    assert rows[0]["estimator_member_predictions"] is None
    assert rows[1]["estimator_member_predictions"] == pytest.approx(
        [0.7, 0.8]
    )
    assert "estimator_prediction" not in rows[0]["scorer_metadata"]


def test_poise_validation_dump_rejects_estimator_row_misalignment(tmp_path):
    trainer = object.__new__(RayPOISETrainer)
    trainer.global_steps = 40
    trainer._poise_validation_dump_rows = [
        {
            "domain": "math",
            "prompt_id": 0,
            "prediction": 0.25,
            "reward": 0.0,
            "loo_target": 1.0,
        }
    ]

    with pytest.raises(
        ValueError,
        match="estimator rows and generation dump rows",
    ):
        trainer._dump_generations(
            inputs=["prompt", "prompt"],
            outputs=["failure", "success"],
            scores=[0.0, 1.0],
            reward_extra_infos_dict={"reward": [0.0, 1.0]},
            dump_path=str(tmp_path),
            record_type="validation",
        )


def test_domain_estimator_updates_are_independent(monkeypatch):
    bank = object.__new__(DomainEstimatorBank)
    bank.states = {
        domain: DomainEstimatorState(estimator=f"initial-{domain}")
        for domain in DOMAIN_NAMES
    }
    bank.warmup_steps = 1
    bank.buffer_max_rows = {domain: 2 for domain in DOMAIN_NAMES}
    bank.feature_config = object()
    bank.fit_config = object()
    fit_calls = []

    def fake_fit_estimator(*, rows, feature_config, fit_config):
        fit_calls.append([row["domain"] for row in rows])
        return f"fitted-{rows[0]['domain']}", {"rows": float(len(rows))}

    monkeypatch.setattr(
        "poise.domain_estimator_bank.fit_estimator",
        fake_fit_estimator,
    )
    metrics = bank.update(
        {
            "math": [{"domain": "math"}],
            "code": [{"domain": "code"}],
            "other": [],
        }
    )

    assert fit_calls == [["math"], ["code"]]
    assert bank.states["math"].estimator == "fitted-math"
    assert bank.states["code"].estimator == "fitted-code"
    assert bank.states["other"].estimator == "initial-other"
    assert bank.states["other"].observed_steps == 0
    assert metrics["poise/estimator/other/refit"] == 0.0


def test_domain_estimator_buffers_trim_complete_prompt_groups_by_row_limit():
    bank = object.__new__(DomainEstimatorBank)
    bank.states = {
        domain: DomainEstimatorState(estimator=f"initial-{domain}")
        for domain in DOMAIN_NAMES
    }
    bank.warmup_steps = 100
    bank.buffer_max_rows = {"math": 4, "code": 2, "other": 2}
    bank.feature_config = object()
    bank.fit_config = object()

    def rows(domain, *uids):
        return [
            {"domain": domain, "uid": uid, "rollout": rollout}
            for uid in uids
            for rollout in range(2)
        ]

    bank.update(
        {
            "math": rows("math", "math-old", "math-middle"),
            "code": rows("code", "code-old"),
            "other": [],
        }
    )
    metrics = bank.update(
        {
            "math": rows("math", "math-new"),
            "code": rows("code", "code-new"),
            "other": rows("other", "other-new"),
        }
    )

    assert [
        row["uid"] for row in bank.states["math"].flattened_rows()
    ] == ["math-middle", "math-middle", "math-new", "math-new"]
    assert [
        row["uid"] for row in bank.states["code"].flattened_rows()
    ] == ["code-new", "code-new"]
    assert len(bank.states["other"].flattened_rows()) == 2
    assert metrics["poise/estimator/math/buffer_rows"] == 4.0
    assert metrics["poise/estimator/math/buffer_max_rows"] == 4.0


def test_bootstrap_recent_buffer_keeps_newest_complete_prompt_groups():
    buffer = RecentDomainBuffer(
        {"math": 4, "code": 2, "other": 2}
    )

    def rows(domain, *uids):
        return [
            {"domain": domain, "uid": uid, "rollout": rollout}
            for uid in uids
            for rollout in range(2)
        ]

    buffer.append(
        {
            "math": rows("math", "old", "middle"),
            "code": rows("code", "code-old"),
            "other": [],
        }
    )
    buffer.append(
        {
            "math": rows("math", "new"),
            "code": rows("code", "code-new"),
            "other": rows("other", "other-new"),
        }
    )

    buffered = buffer.rows_by_domain()
    assert [row["uid"] for row in buffered["math"]] == [
        "middle",
        "middle",
        "new",
        "new",
    ]
    assert [row["uid"] for row in buffered["code"]] == [
        "code-new",
        "code-new",
    ]
    assert buffer.row_counts == {"math": 4, "code": 2, "other": 2}


def test_bootstrap_rloo_collects_equal_reward_groups():
    batch = DataProto.from_single_dict(
        {
            "responses": torch.zeros((4, 1), dtype=torch.long),
            "uid": np.asarray(["mixed", "mixed", "zero", "zero"]),
        }
    )
    trainer = object.__new__(RayPOISETrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(n=2)
        )
    )
    trainer._poise_runtime = SimpleNamespace(
        fit_config=SimpleNamespace(
            target_mode="other_rollout_correctness"
        )
    )
    trainer._build_poise_feature_rows = lambda **kwargs: (
        ["math", "math", "other", "other"],
        [
            {
                "domain": domain,
                "prompt_hidden": np.zeros(1),
                "response_hidden": np.zeros(1),
                "response_features": {},
            }
            for domain in ("math", "math", "other", "other")
        ],
    )

    rows, _ = trainer._compute_bootstrap_rloo_advantages(
        batch=batch,
        reward_sums=torch.tensor([1.0, 0.0, 0.25, 0.25]),
        entropies=torch.zeros((4, 1)),
    )

    torch.testing.assert_close(
        batch.batch["poise_raw_advantages"],
        torch.tensor([1.0, -1.0, 0.0, 0.0]),
    )
    assert [row["target"] for row in rows["math"]] == [0.0, 1.0]
    assert [row["uid"] for row in rows["other"]] == ["zero", "zero"]


def test_fast_pca_legacy_runtime():
    projection = FastPCA()
    projection.components_ = np.asarray([[1.0, -1.0]], dtype=np.float32)
    projection.mean_ = np.asarray([0.5, 0.5], dtype=np.float32)

    result = projection.transform([[2.0, 1.0]])

    np.testing.assert_allclose(result, [[1.0]])


def test_feature_builder_supports_builtin_and_derived_scalars():
    config = FeatureBuilderConfig.from_dict(
        {
            "prompt_hidden": {
                "input_field": "prompt_hidden_states",
                "layer_index": 1,
                "pooling": {"type": "last_n_mean", "n": 2},
            },
            "response_hidden": {
                "input_field": "response_hidden_states",
                "layer_index": 1,
                "pooling": {"type": "last_n_mean", "n": 2},
            },
            "rollout_scalars": {
                "scalar_keys": ["output_length", "has_complete_answer"],
                "derived_scalar_keys": ["answer_ratio"],
            },
        }
    )
    builder = SingleTrajectoryFeatureBuilder(config)

    _, _, features = builder.build_inputs(
        prompt_hidden=[1.0, 2.0],
        response_hidden=[3.0, 4.0],
        generated_text="<think>x</think><answer>yes</answer>",
        response_ids=[1, 2, 3, 4],
        tokenizer=None,
        rollout_features={},
    )

    assert features == {
        "output_length": 4.0,
        "has_complete_answer": 1.0,
        "answer_ratio": 0.0,
    }


def test_capture_spec_supports_distinct_prompt_and_response_layers():
    bank = object.__new__(DomainEstimatorBank)
    bank.feature_config = FeatureBuilderConfig.from_dict(
        {
            "prompt_hidden": {
                "input_field": "prompt_hidden_states",
                "layer_index": 3,
                "pooling": {"type": "last_n_mean", "n": 2},
            },
            "response_hidden": {
                "input_field": "response_hidden_states",
                "layer_index": 7,
                "pooling": {"type": "last_n_mean", "n": 4},
            },
            "rollout_scalars": {"scalar_keys": []},
        }
    )

    assert bank.capture_spec == {
        "prompt_layer_index": 3,
        "response_layer_index": 7,
        "prompt_pool_n": 2,
        "response_pool_n": 4,
    }


def test_hidden_capture_pools_prompt_and_response_from_distinct_layers():
    actor = object.__new__(DataParallelPPOActor)
    prompt_layer_hidden = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0],
          [5.0, 50.0], [6.0, 60.0]]]
    )
    response_layer_hidden = prompt_layer_hidden + 100.0

    capture = actor._build_hidden_capture(
        prompt_full_hidden=prompt_layer_hidden,
        response_full_hidden=response_layer_hidden,
        attention_mask=torch.ones((1, 6), dtype=torch.long),
        response_mask=torch.ones((1, 3), dtype=torch.long),
        response_length=3,
        prompt_pool_n=2,
        response_pool_n=2,
        response_ids=torch.tensor([[11, 12, 13]]),
        think_end_token_ids=[],
    )

    torch.testing.assert_close(
        capture["prompt_hidden"], torch.tensor([[2.5, 25.0]])
    )
    torch.testing.assert_close(
        capture["response_hidden"], torch.tensor([[105.5, 155.0]])
    )


def test_actor_forward_hooks_each_requested_hidden_layer(monkeypatch):
    monkeypatch.setattr(
        "verl.workers.actor.dp_actor.logprobs_from_logits",
        lambda logits, labels: torch.log_softmax(logits, dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1),
    )

    class AddLayer(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = float(offset)

        def forward(self, hidden):
            return (hidden + self.offset,)

    class LayerStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([AddLayer(10), AddLayer(20)])

    class FakeActorModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = LayerStack()

        def forward(self, input_ids, **_kwargs):
            hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 2)
            for layer in self.model.layers:
                hidden = layer(hidden)[0]
            logits = torch.zeros((*input_ids.shape, 32), dtype=torch.float32)
            logits[..., 0] = hidden[..., 0]
            return SimpleNamespace(logits=logits)

    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = FakeActorModule()
    actor.device_name = "cpu"
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.config = SimpleNamespace(entropy_checkpointing=False)
    actor._capture_layer_cache = {}

    _, _, capture = actor._forward_micro_batch(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "position_ids": torch.arange(6).reshape(1, 6),
            "responses": torch.tensor([[4, 5, 6]]),
            "response_mask": torch.ones((1, 3), dtype=torch.long),
        },
        temperature=1.0,
        calculate_entropy=True,
        hidden_capture_spec={
            "prompt_layer_index": 0,
            "response_layer_index": 1,
            "prompt_pool_n": 2,
            "response_pool_n": 2,
            "think_end_token_ids": [],
        },
    )

    torch.testing.assert_close(
        capture["prompt_hidden"], torch.tensor([[12.5, 12.5]])
    )
    torch.testing.assert_close(
        capture["response_hidden"], torch.tensor([[35.5, 35.5]])
    )
    assert not actor.actor_module.model.layers[0]._forward_hooks
    assert not actor.actor_module.model.layers[1]._forward_hooks


def test_olmo_probe_config_drives_capture_and_probe_fit():
    repo_root = Path(__file__).resolve().parents[1]
    runtime = DomainEstimatorBank.runtime(
        feature_config_path=str(
            repo_root
            / "src/poise/config/feature_builder_config_olmo3_7b.json"
        ),
        fit_config_path=str(
            repo_root
            / "src/poise/config/estimator_fit_config_olmo3_7b.json"
        ),
        warmup_steps=8,
        buffer_max_rows={domain: 512 for domain in DOMAIN_NAMES},
    )

    assert runtime.capture_spec == {
        "prompt_layer_index": 7,
        "response_layer_index": 7,
        "prompt_pool_n": 32,
        "response_pool_n": 32,
        "layer_index": 7,
    }
    assert runtime.fit_config.prompt_hidden_pca_dim == 16
    assert runtime.fit_config.response_hidden_pca_dim == 128
    assert runtime.fit_config.alpha == 1000.0

    rng = np.random.default_rng(42)
    rows = [
        {
            "prompt_hidden": rng.standard_normal(160).astype(np.float32),
            "response_hidden": rng.standard_normal(160).astype(np.float32),
            "response_features": {
                key: float(rng.random())
                for key in runtime.feature_config.rollout_scalars.scalar_keys
            },
            "target": float(index % 2),
        }
        for index in range(130)
    ]

    estimator, _ = fit_estimator(
        rows=rows,
        feature_config=runtime.feature_config,
        fit_config=runtime.fit_config,
    )

    assert estimator.prompt_hidden_projection.n_components_ == 16
    assert estimator.response_hidden_projection.n_components_ == 128
    assert estimator.config.model.feature_dim == 147
    assert estimator.estimator.named_steps["model"].alpha == 1000.0


def _checkpoint_test_feature_config():
    return FeatureBuilderConfig.from_dict(
        {
            "prompt_hidden": {
                "input_field": "prompt_hidden_states",
                "layer_index": 1,
                "pooling": {"type": "last_n_mean", "n": 1},
            },
            "response_hidden": {
                "input_field": "response_hidden_states",
                "layer_index": 1,
                "pooling": {"type": "last_n_mean", "n": 1},
            },
            "rollout_scalars": {
                "scalar_keys": ["score"],
            },
        }
    )


def _checkpoint_test_estimator(label):
    return SingleTrajectoryEstimator(
        config=EstimatorConfig(
            prompt_hidden_projection=ProjectionConfig(
                type=None,
                input_dim=None,
                output_dim=None,
            ),
            response_hidden_projection=ProjectionConfig(
                type=None,
                input_dim=None,
                output_dim=None,
            ),
            response_feature_keys=("score",),
            derived_response_feature_keys=(),
            model=EstimatorModelConfig(
                alpha=0.01,
                clip_min=0.0,
                clip_max=1.0,
                feature_dim=0,
            ),
        ),
        estimator=label,
        prompt_hidden_projection=None,
        response_hidden_projection=None,
    )


def test_bootstrap_fits_each_domain_from_partial_recent_buffers(
    monkeypatch,
    tmp_path,
):
    feature_path = tmp_path / "features.json"
    feature_path.write_text(
        json.dumps(
            {
                "prompt_hidden": {
                    "input_field": "prompt_hidden_states",
                    "layer_index": 1,
                    "pooling": {"type": "last_n_mean", "n": 1},
                },
                "response_hidden": {
                    "input_field": "response_hidden_states",
                    "layer_index": 1,
                    "pooling": {"type": "last_n_mean", "n": 1},
                },
                "rollout_scalars": {"scalar_keys": ["score"]},
            }
        )
    )
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(
        json.dumps(
            {
                "prompt_hidden_pca_dim": 0,
                "response_hidden_pca_dim": 0,
                "target_mode": "other_rollout_correctness",
            }
        )
    )
    fit_sizes = []

    def fake_fit_estimator(*, rows, feature_config, fit_config):
        del feature_config, fit_config
        fit_sizes.append(len(rows))
        return _checkpoint_test_estimator("fitted"), {
            "train_rows": float(len(rows))
        }

    monkeypatch.setattr(
        "poise.domain_estimator_bank.fit_estimator",
        fake_fit_estimator,
    )

    def rows(domain, count):
        return [
            {
                "domain": domain,
                "uid": f"{domain}-{index // 2}",
                "target": float(index % 2),
            }
            for index in range(count)
        ]

    bank, metrics = DomainEstimatorBank.from_rows(
        rows_by_domain={
            "math": rows("math", 4),
            "code": rows("code", 2),
            "other": rows("other", 6),
        },
        feature_config_path=str(feature_path),
        fit_config_path=str(fit_path),
        warmup_steps=4,
        buffer_max_rows={"math": 20, "code": 20, "other": 20},
        observed_steps=24,
    )

    assert fit_sizes == [4, 2, 6]
    assert bank.initial_model_paths == {}
    assert {
        domain: len(bank.states[domain].flattened_rows())
        for domain in DOMAIN_NAMES
    } == {"math": 4, "code": 2, "other": 6}
    assert all(
        bank.states[domain].observed_steps == 24
        for domain in DOMAIN_NAMES
    )
    assert metrics["poise/estimator/code/bootstrap_rows"] == 2.0


def test_multi_estimator_exposes_combined_and_child_predictions():
    class LinearChild:
        def __init__(self, scale):
            self.scale = scale
            self.n_features_in_ = 3

        def predict(self, values):
            values = np.asarray(values, dtype=np.float32)
            return values[:, 0] * self.scale

    estimator = SingleTrajectoryEstimator(
        config=EstimatorConfig(
            prompt_hidden_projection=ProjectionConfig(
                type=None,
                input_dim=None,
                output_dim=None,
            ),
            response_hidden_projection=ProjectionConfig(
                type=None,
                input_dim=None,
                output_dim=None,
            ),
            response_feature_keys=("score",),
            derived_response_feature_keys=(),
            model=EstimatorModelConfig(
                alpha=0.01,
                clip_min=0.0,
                clip_max=1.0,
                feature_dim=3,
            ),
        ),
        estimator=MultiEstimatorRegressor(
            estimators=[LinearChild(0.2), LinearChild(0.6)],
            combine_method="mean",
            stacker=None,
            clip_min=0.0,
            clip_max=1.0,
            alpha=0.01,
        ),
        prompt_hidden_projection=None,
        response_hidden_projection=None,
    )

    combined, members = estimator.predict_value_with_members(
        prompt_hidden=[1.0],
        response_hidden=[2.0],
        response_features={"score": 0.5},
    )

    assert combined == pytest.approx(0.4)
    assert members == pytest.approx([0.2, 0.6])


def test_multi_estimator_fit_reports_post_fit_member_diagnostics():
    feature_config = _checkpoint_test_feature_config()
    fit_config = EstimatorFitConfig(
        prompt_hidden_pca_dim=0,
        response_hidden_pca_dim=0,
        child_fit_configs=[{"alpha": 0.01}, {"alpha": 1.0}],
    )
    rows = [
        {
            "prompt_hidden": np.asarray([value], dtype=np.float32),
            "response_hidden": np.asarray([1.0 - value], dtype=np.float32),
            "response_features": {"score": value},
            "target": value,
        }
        for value in (0.0, 0.25, 0.75, 1.0)
    ]

    estimator, metrics = fit_estimator(
        rows=rows,
        feature_config=feature_config,
        fit_config=fit_config,
    )

    assert isinstance(estimator.estimator, MultiEstimatorRegressor)
    assert metrics["member_count"] == 2.0
    assert "train_rmse" in metrics
    assert "train_constant_brier" in metrics
    assert "train_brier_skill" in metrics
    assert metrics["train_brier_skill_defined"] == 1.0
    assert "train_bias" in metrics
    assert "train_pearson" in metrics
    assert "member_0/train_mae" in metrics
    assert "member_1/train_pearson" in metrics


def _checkpoint_test_bank():
    bank = object.__new__(DomainEstimatorBank)
    bank.feature_config = _checkpoint_test_feature_config()
    bank.warmup_steps = 4
    bank.buffer_max_rows = {domain: 8 for domain in DOMAIN_NAMES}
    bank.states = {
        domain: DomainEstimatorState(
            estimator=_checkpoint_test_estimator(f"trained-{domain}"),
            buffer_steps=[
                [
                    {
                        "domain": domain,
                        "uid": f"{domain}-prompt",
                        "target": 0.5,
                    }
                ]
            ],
            observed_steps=5,
            retrain_count=2,
        )
        for domain in DOMAIN_NAMES
    }
    return bank


def test_domain_estimator_checkpoint_round_trip_restores_training_state(tmp_path):
    bank = _checkpoint_test_bank()
    bank.save(tmp_path)

    restored = _checkpoint_test_bank()
    restored.states = {
        domain: DomainEstimatorState(
            estimator=_checkpoint_test_estimator(f"initial-{domain}")
        )
        for domain in DOMAIN_NAMES
    }

    assert restored.load(tmp_path)
    for domain in DOMAIN_NAMES:
        state = restored.states[domain]
        assert state.estimator.estimator == f"trained-{domain}"
        assert state.observed_steps == 5
        assert state.retrain_count == 2
        assert state.flattened_rows()[0]["uid"] == f"{domain}-prompt"


def test_poise_checkpoint_saves_estimator_before_latest_pointer(
    monkeypatch,
    tmp_path,
):
    events = []

    class RecordingBank:
        def save(self, checkpoint_dir):
            events.append(("estimator", checkpoint_dir))

    trainer = object.__new__(RayPOISETrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            default_local_dir=str(tmp_path),
            poise=SimpleNamespace(
                bootstrap=SimpleNamespace(from_scratch=False)
            ),
        )
    )
    trainer.global_steps = 12
    trainer._poise_bank = RecordingBank()
    monkeypatch.setattr(
        RayPPOTrainer,
        "_save_checkpoint",
        lambda self: events.append(("base", self.global_steps)),
    )

    trainer._save_checkpoint()

    assert events == [
        ("estimator", str(tmp_path / "global_step_12")),
        ("base", 12),
    ]


def test_poise_estimator_update_snapshot_uses_independent_ten_step_cadence(
    tmp_path,
):
    saved_paths = []

    class RecordingBank:
        def save(self, checkpoint_dir):
            saved_paths.append(checkpoint_dir)

        def adaptive_training_data_payload(self, *, global_step):
            return {
                "schema_version": 2,
                "global_step": global_step,
                "retrain_steps": 20,
                "retrain_count": 5,
                "retrain_steps_by_domain": {
                    domain: 20 for domain in DOMAIN_NAMES
                },
                "retrain_count_by_domain": {
                    domain: 5 for domain in DOMAIN_NAMES
                },
                "refit_domains_this_step": list(DOMAIN_NAMES),
                "estimator_trained_from_saved_data": True,
                "estimator_files": {
                    domain: f"poise_{domain}_estimator.joblib"
                    for domain in DOMAIN_NAMES
                },
                "buffer_rows": 6,
                "buffer_steps": 1,
                "buffer_step_row_counts": [6],
                "buffer_max_rows_by_domain": {
                    domain: 8 for domain in DOMAIN_NAMES
                },
                "domain_summaries": {
                    domain: {"buffer_rows": 2} for domain in DOMAIN_NAMES
                },
                "train_domains": list(DOMAIN_NAMES) * 2,
                "train_prompt_hidden_rows": [],
                "train_response_hidden_rows": [],
                "train_response_feature_rows": [],
                "train_targets": [],
                "train_rewards": [],
                "train_group_keys": [],
                "train_group_types": [],
            }

    trainer = object.__new__(RayPOISETrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "poise": {
                    "estimator": {
                        "save_freq": 10,
                        "online_output_dir": None,
                    }
                },
            }
        }
    )
    trainer._poise_phase = "poise"
    trainer._poise_bank = RecordingBank()

    trainer.global_steps = 16
    assert not trainer._should_save_estimator_update()
    trainer.global_steps = 20
    assert trainer._should_save_estimator_update()

    snapshot_dir = trainer._save_estimator_update_snapshot()

    expected_dir = tmp_path / "adaptive_estimator_updates" / "global_step_20"
    assert snapshot_dir == str(expected_dir)
    assert saved_paths == [str(expected_dir)]
    assert (
        tmp_path
        / "adaptive_estimator_updates"
        / "latest_adaptive_estimator_update.txt"
    ).read_text() == "20"
    training_data = torch.load(
        expected_dir / "poise_adaptive_estimator_training_data.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert training_data["global_step"] == 20
    metadata = json.loads(
        (expected_dir / "poise_adaptive_estimator_update.meta.json").read_text()
    )
    assert metadata["global_step"] == 20
    assert metadata["training_data_saved"]
    assert set(metadata["estimator_model_paths"]) == set(DOMAIN_NAMES)

    trainer._poise_phase = "bootstrap"
    trainer.global_steps = 30
    assert not trainer._should_save_estimator_update()


def test_domain_estimator_training_data_snapshot_has_flat_compatibility_fields():
    bank = _checkpoint_test_bank()
    for domain_index, domain in enumerate(DOMAIN_NAMES):
        row = bank.states[domain].buffer_steps[0][0]
        row.update(
            {
                "global_step": 20,
                "prompt_hidden": np.asarray([domain_index, 1.0], dtype=np.float32),
                "response_hidden": np.asarray([2.0, domain_index], dtype=np.float32),
                "response_features": {"score": float(domain_index)},
                "reward": float(domain_index == 0),
            }
        )

    payload = bank.adaptive_training_data_payload(global_step=20)

    assert payload["global_step"] == 20
    assert payload["buffer_rows"] == 3
    assert payload["buffer_steps"] == 1
    assert payload["buffer_step_row_counts"] == [3]
    assert payload["train_domains"] == list(DOMAIN_NAMES)
    assert len(payload["train_prompt_hidden_rows"]) == 3
    assert len(payload["train_response_hidden_rows"]) == 3
    assert len(payload["train_response_feature_rows"]) == 3
    assert payload["train_targets"] == [0.5, 0.5, 0.5]
    assert payload["train_group_types"] == ["all1", "all0", "all0"]
    assert payload["train_buffer_sources"] == ["recent"] * 3
    assert payload["recent_buffer_rows"] == 3


def test_poise_estimator_update_snapshot_writes_training_data_and_metadata(
    tmp_path,
):
    bank = _checkpoint_test_bank()
    for domain_index, domain in enumerate(DOMAIN_NAMES):
        bank.states[domain].buffer_steps[0][0].update(
            {
                "global_step": 20,
                "prompt_hidden": np.asarray([domain_index, 1.0], dtype=np.float32),
                "response_hidden": np.asarray([2.0, domain_index], dtype=np.float32),
                "response_features": {"score": float(domain_index)},
                "reward": float(domain_index == 0),
            }
        )

    trainer = object.__new__(RayPOISETrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "poise": {
                    "estimator": {
                        "save_freq": 10,
                        "online_output_dir": None,
                    }
                },
            }
        }
    )
    trainer.global_steps = 20
    trainer._poise_phase = "poise"
    trainer._poise_bank = bank

    snapshot_dir = Path(trainer._save_estimator_update_snapshot())

    assert {path.name for path in snapshot_dir.iterdir()} == {
        "poise_math_estimator.joblib",
        "poise_code_estimator.joblib",
        "poise_other_estimator.joblib",
        "poise_domain_estimator_state.pt",
        "poise_adaptive_estimator_training_data.pt",
        "poise_adaptive_estimator_update.meta.json",
    }
    training_data = torch.load(
        snapshot_dir / "poise_adaptive_estimator_training_data.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert training_data["buffer_rows"] == 3
    metadata = json.loads(
        (snapshot_dir / "poise_adaptive_estimator_update.meta.json").read_text()
    )
    assert metadata["snapshot_saved"]
    assert metadata["training_data_saved"]


def test_scratch_phase_checkpoint_restores_recent_buffer(
    tmp_path,
):
    config = OmegaConf.create(
        {
            "data": {"seed": 42},
            "trainer": {
                "poise": {
                    "bootstrap": {
                        "steps": 16,
                    }
                }
            },
        }
    )
    limits = {"math": 4, "code": 2, "other": 2}
    checkpoint_dir = tmp_path / "global_step_12"
    trainer = object.__new__(RayPOISETrainer)
    trainer.config = config
    trainer.global_steps = 12
    trainer._poise_phase = "bootstrap"
    trainer._poise_runtime = SimpleNamespace(buffer_max_rows=limits)
    trainer._bootstrap_buffer = RecentDomainBuffer(limits)
    trainer._bootstrap_buffer.append(
        {
            "math": [
                {"domain": "math", "uid": "prompt", "rollout": 0},
                {"domain": "math", "uid": "prompt", "rollout": 1},
            ],
            "code": [],
            "other": [],
        }
    )
    trainer._save_phase_state(str(checkpoint_dir))

    state_path = checkpoint_dir / trainer._PHASE_STATE_FILENAME
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    assert set(payload) == {"schema_version", "phase", "bootstrap_steps", "bootstrap_mode", "buffer"}
    assert payload["bootstrap_mode"] == "rloo"

    restored = object.__new__(RayPOISETrainer)
    restored.config = config
    restored.global_steps = 12
    restored._poise_runtime = SimpleNamespace(buffer_max_rows=limits)
    restored._bootstrap_buffer = RecentDomainBuffer(limits)
    restored._resolve_resume_checkpoint_dir = lambda: str(checkpoint_dir)
    restored._restore_scratch_checkpoint()

    assert restored._poise_phase == "bootstrap"
    assert restored._bootstrap_buffer.row_counts["math"] == 2

    payload["bootstrap_mode"] = "unsupported"
    torch.save(payload, state_path)
    with pytest.raises(ValueError, match="requires an RLOO bootstrap checkpoint"):
        restored._restore_scratch_checkpoint()


def test_resume_rejects_actor_checkpoint_without_estimator_state(tmp_path):
    class MissingBank:
        def load(self, checkpoint_dir):
            return False

    trainer = object.__new__(RayPOISETrainer)
    trainer.global_steps = 12
    trainer._poise_bank = MissingBank()
    trainer._resolve_resume_checkpoint_dir = lambda: str(tmp_path)

    with pytest.raises(FileNotFoundError, match="required POISE estimator state"):
        trainer._restore_estimator_checkpoint()


def test_online_estimator_metrics_are_domain_and_member_specific():
    class MetricBank:
        fit_config = SimpleNamespace(target_mode="other_rollout_correctness")

        @staticmethod
        def estimator_for(domain):
            del domain
            return SimpleNamespace(
                config=SimpleNamespace(
                    model=SimpleNamespace(clip_min=0.0, clip_max=1.0)
                )
            )

    trainer = object.__new__(RayPOISETrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            poise=SimpleNamespace(
                estimator=SimpleNamespace(group_size=2)
            )
        )
    )
    trainer._poise_bank = MetricBank()
    metrics = trainer._estimator_online_metrics(
        {
            "math": [
                {
                    "uid": "prompt",
                    "prediction": 0.2,
                    "target": 0.0,
                    "reward": 1.0,
                    "cross_baseline": 0.8,
                    "advantage": 0.2,
                    "member_predictions": [0.1, 0.3],
                },
                {
                    "uid": "prompt",
                    "prediction": 0.8,
                    "target": 1.0,
                    "reward": 0.0,
                    "cross_baseline": 0.2,
                    "advantage": -0.2,
                    "member_predictions": [0.9, 0.7],
                },
            ],
            "code": [],
            "other": [],
        }
    )

    assert metrics["poise/online/math/target_mae"] == pytest.approx(0.2)
    assert metrics["poise/online/math/target_rmse"] == pytest.approx(0.2)
    assert metrics["poise/online/math/constant_brier"] == pytest.approx(0.25)
    assert metrics["poise/online/math/brier_skill"] == pytest.approx(0.84)
    assert metrics["poise/online/math/brier_skill_defined"] == 1.0
    assert abs(metrics["poise/online/math/target_bias"]) < 1e-7
    assert metrics["poise/online/math/target_pearson"] == pytest.approx(1.0)
    assert metrics["poise/online/math/pairwise_sign_acc"] == pytest.approx(1.0)
    assert metrics["poise/online/math/member_count"] == 2.0
    assert metrics[
        "poise/online/math/member_0/target_mae"
    ] == pytest.approx(0.1)
    assert metrics[
        "poise/online/math/cross_rollout/baseline_vs_reward/mae"
    ] == pytest.approx(0.2)


def test_prompt_reward_log_contains_per_domain_estimator_values(tmp_path):
    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return " ".join(str(token_id) for token_id in token_ids)

    raw_prompts = np.empty(4, dtype=object)
    raw_prompts[:] = [
        [{"role": "user", "content": "math prompt"}],
        [{"role": "user", "content": "math prompt"}],
        [{"role": "user", "content": "code prompt"}],
        [{"role": "user", "content": "code prompt"}],
    ]
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.tensor(
                [
                    [0, 10, 11],
                    [0, 10, 11],
                    [0, 20, 21],
                    [0, 20, 21],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                ]
            ),
            "uid": np.asarray(["math-uid", "math-uid", "code-uid", "code-uid"]),
            "data_source": np.asarray(
                [
                    "math__combined",
                    "math__combined",
                    "codegen__taco",
                    "codegen__taco",
                ]
            ),
            "raw_prompt": raw_prompts,
        }
    )
    batch.batch["poise_value_predictions"] = torch.tensor(
        [0.2, 0.8, 0.3, 0.7]
    )
    batch.batch["poise_baselines"] = torch.tensor([0.8, 0.2, 0.7, 0.3])
    batch.batch["poise_targets"] = torch.tensor([0.0, 1.0, 1.0, 0.0])
    batch.batch["poise_raw_advantages"] = torch.tensor([0.2, 0.8, 0.3, -0.3])
    member_predictions = np.empty(4, dtype=object)
    member_predictions[:] = [
        [0.1, 0.3],
        [0.9, 0.7],
        [0.2, 0.4],
        [0.8, 0.6],
    ]
    batch.non_tensor_batch["poise_member_value_predictions"] = member_predictions

    trainer = object.__new__(RayPOISETrainer)
    trainer.tokenizer = Tokenizer()
    trainer.global_steps = 7
    accumulator = {}
    order = []
    trainer._accumulate_prompt_reward_log_rows(
        accumulator=accumulator,
        order=order,
        batch=batch,
        reward_sums=torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    filename = trainer._dump_prompt_reward_log(
        output_dir=str(tmp_path),
        accumulator=accumulator,
        order=order,
        rollout_repeat=2,
        final_train_batch_rows=4,
    )

    with open(filename, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["prompt_count"] == 2
    assert payload["schema_version"] == 7
    assert payload["trajectory_count"] == 4
    assert payload["estimator_member_count"] == 2
    assert payload["estimator_member_counts_by_domain"] == {
        "math": 2,
        "code": 2,
    }
    assert payload["records"][0]["domain"] == "math"
    assert payload["records"][0]["rollout_rewards"] == [1.0, 0.0]
    assert payload["records"][0]["estimator_value_predictions"] == pytest.approx(
        [0.2, 0.8]
    )
    assert payload["records"][0]["estimator_online_mae"] == pytest.approx(0.2)
    assert payload["records"][0][
        "estimator_online_constant_brier"
    ] == pytest.approx(0.25)
    assert payload["records"][0][
        "estimator_online_brier_skill"
    ] == pytest.approx(0.84)
    assert payload["records"][0][
        "estimator_online_brier_skill_defined"
    ] == 1.0
    np.testing.assert_allclose(
        payload["records"][0]["estimator_member_value_predictions"],
        [[0.1, 0.3], [0.9, 0.7]],
    )
    assert payload["domain_summaries"]["math"]["online_rmse"] == pytest.approx(
        0.2
    )
    assert payload["domain_summaries"]["math"][
        "online_brier_skill"
    ] == pytest.approx(0.84)
    assert payload["domain_summaries"]["math"][
        "estimator_member_count"
    ] == 2
    assert payload["records"][1]["data_source"] == "codegen__taco"
    assert (tmp_path / "latest_prompt_reward_log.txt").read_text() == "7"
