"""Pure tensor helpers for POISE targets and leave-one-out baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import torch


SUPPORTED_TARGET_MODES = frozenset({"pair_average", "other_rollout_correctness"})


def regression_diagnostic_values(
    *,
    predictions: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return finite regression diagnostics without imposing a metric prefix."""
    prediction_values = np.asarray(predictions, dtype=np.float32).reshape(-1)
    target_values = np.asarray(targets, dtype=np.float32).reshape(-1)
    if prediction_values.size != target_values.size:
        raise ValueError(
            "Regression diagnostics require matching prediction and target rows: "
            f"{prediction_values.size} != {target_values.size}"
        )
    if prediction_values.size == 0:
        return {}
    if not np.isfinite(prediction_values).all() or not np.isfinite(
        target_values
    ).all():
        raise ValueError("Regression diagnostics require finite values")

    errors = prediction_values - target_values
    squared_errors = np.square(errors)
    target_mean = float(target_values.mean())
    brier_score = float(np.mean(squared_errors))
    constant_brier = float(
        np.mean(np.square(target_values - target_mean))
    )
    brier_skill_defined = constant_brier > 1e-12
    brier_skill = (
        float(1.0 - brier_score / constant_brier)
        if brier_skill_defined
        else 0.0
    )
    prediction_quantiles = np.quantile(
        prediction_values,
        [0.1, 0.5, 0.9],
    )
    target_quantiles = np.quantile(target_values, [0.1, 0.5, 0.9])
    prediction_std = float(prediction_values.std())
    target_std = float(target_values.std())
    if (
        prediction_values.size > 1
        and prediction_std > 1e-8
        and target_std > 1e-8
    ):
        pearson = float(np.corrcoef(prediction_values, target_values)[0, 1])
        if not np.isfinite(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    return {
        "rows": float(prediction_values.size),
        "prediction_mean": float(prediction_values.mean()),
        "prediction_std": prediction_std,
        "prediction_p10": float(prediction_quantiles[0]),
        "prediction_p50": float(prediction_quantiles[1]),
        "prediction_p90": float(prediction_quantiles[2]),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_p10": float(target_quantiles[0]),
        "target_p50": float(target_quantiles[1]),
        "target_p90": float(target_quantiles[2]),
        "target_mae": float(np.mean(np.abs(errors))),
        "target_rmse": float(np.sqrt(brier_score)),
        # For probability predictions, target_rmse**2 is the raw Brier
        # score. Log only the non-redundant constant reference and skill.
        "constant_brier": constant_brier,
        "brier_skill": brier_skill,
        "brier_skill_defined": float(brier_skill_defined),
        "target_bias": float(errors.mean()),
        "target_pearson": pearson,
    }


def compute_group_targets_and_cross_baselines(
    *,
    reward_sums: torch.Tensor,
    value_predictions: torch.Tensor,
    uid_to_indices: Mapping[str, Sequence[int]],
    group_size: int,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute estimator labels and action-independent cross-rollout baselines.

    ``value_predictions[i]`` is produced from rollout ``i``.  The policy
    baseline for rollout ``i`` is the mean prediction from its sibling
    rollouts.  With two rollouts this is the original POISE sibling swap.
    """
    if target_mode not in SUPPORTED_TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(SUPPORTED_TARGET_MODES)}, got {target_mode!r}"
        )
    if group_size < 2:
        raise ValueError(f"POISE requires group_size >= 2, got {group_size}")
    if reward_sums.ndim != 1 or value_predictions.ndim != 1:
        raise ValueError("POISE rewards and predictions must be rank-1 tensors")
    if reward_sums.numel() != value_predictions.numel():
        raise ValueError("POISE rewards and predictions must contain the same number of rows")

    rewards = reward_sums.to(torch.float32)
    predictions = value_predictions.to(device=rewards.device, dtype=torch.float32)
    targets = torch.empty_like(rewards)
    cross_baselines = torch.empty_like(rewards)

    for uid, raw_indices in uid_to_indices.items():
        indices = [int(idx) for idx in raw_indices]
        if len(indices) != group_size:
            raise ValueError(
                f"uid={uid!r} appears {len(indices)} times; expected group_size={group_size}"
            )
        index_tensor = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        group_rewards = rewards[index_tensor]
        group_predictions = predictions[index_tensor]
        denominator = float(group_size - 1)

        cross_baselines[index_tensor] = (
            group_predictions.sum() - group_predictions
        ) / denominator
        if target_mode == "other_rollout_correctness":
            targets[index_tensor] = (group_rewards.sum() - group_rewards) / denominator
        else:
            targets[index_tensor] = group_rewards.mean()

    return targets, cross_baselines


def cross_rollout_pairwise_sign_matches(
    *,
    reward_sums: torch.Tensor,
    value_predictions: torch.Tensor,
    uid_to_indices: Mapping[str, Sequence[int]],
    group_size: int,
    target_mode: str,
) -> list[float]:
    """Compare within-group prediction ordering with reward ordering."""
    if target_mode not in SUPPORTED_TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(SUPPORTED_TARGET_MODES)}, "
            f"got {target_mode!r}"
        )
    rewards = reward_sums.detach().to(torch.float32).cpu()
    predictions = value_predictions.detach().to(torch.float32).cpu()
    matches: list[float] = []
    for raw_indices in uid_to_indices.values():
        indices = [int(index) for index in raw_indices]
        if len(indices) != group_size:
            continue
        for first_index, second_index in combinations(indices, 2):
            if target_mode == "other_rollout_correctness":
                prediction_difference = float(
                    predictions[second_index] - predictions[first_index]
                )
            else:
                prediction_difference = float(
                    predictions[first_index] - predictions[second_index]
                )
            reward_difference = float(
                rewards[first_index] - rewards[second_index]
            )
            matches.append(
                1.0
                if np.sign(prediction_difference)
                == np.sign(reward_difference)
                else 0.0
            )
    return matches


def cross_rollout_diagnostic_metrics(
    *,
    rewards: Sequence[float] | np.ndarray,
    value_predictions: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    cross_baselines: Sequence[float] | np.ndarray,
    advantages: Sequence[float] | np.ndarray,
    group_size: int,
    prefix: str,
) -> dict[str, float]:
    """Build detailed leave-one-out diagnostics for console and W&B."""
    if group_size < 2:
        raise ValueError(f"POISE requires group_size >= 2, got {group_size}")
    arrays = {
        "reward": np.asarray(rewards, dtype=np.float32).reshape(-1),
        "value_prediction": np.asarray(
            value_predictions,
            dtype=np.float32,
        ).reshape(-1),
        "estimator_target": np.asarray(targets, dtype=np.float32).reshape(-1),
        "cross_baseline": np.asarray(
            cross_baselines,
            dtype=np.float32,
        ).reshape(-1),
        "advantage": np.asarray(advantages, dtype=np.float32).reshape(-1),
    }
    row_count = arrays["reward"].size
    if row_count == 0:
        return {}
    mismatched = {
        name: values.size
        for name, values in arrays.items()
        if values.size != row_count
    }
    if mismatched:
        raise ValueError(
            "Cross-rollout diagnostic rows must have identical lengths: "
            f"expected={row_count}, mismatched={mismatched}"
        )
    if row_count % group_size != 0:
        raise ValueError(
            "Cross-rollout diagnostic row count must be divisible by group "
            f"size: rows={row_count}, group_size={group_size}"
        )
    non_finite = {
        name: int((~np.isfinite(values)).sum())
        for name, values in arrays.items()
    }
    if any(non_finite.values()):
        raise ValueError(
            f"Cross-rollout diagnostics received non-finite values: {non_finite}"
        )

    metrics = {
        f"{prefix}/group_size": float(group_size),
        f"{prefix}/num_rows": float(row_count),
        f"{prefix}/num_groups": float(row_count / group_size),
        f"{prefix}/baseline_self_weight": 0.0,
        f"{prefix}/baseline_other_weight": float(1.0 / (group_size - 1)),
    }
    for name, values in arrays.items():
        quantiles = np.quantile(values, [0.1, 0.5, 0.9])
        metrics.update(
            {
                f"{prefix}/{name}/mean": float(values.mean()),
                f"{prefix}/{name}/var": float(values.var()),
                f"{prefix}/{name}/p10": float(quantiles[0]),
                f"{prefix}/{name}/p50": float(quantiles[1]),
                f"{prefix}/{name}/p90": float(quantiles[2]),
            }
        )

    baseline_values = arrays["cross_baseline"]
    reward_values = arrays["reward"]
    baseline_metrics = regression_diagnostic_values(
        predictions=baseline_values,
        targets=reward_values,
    )
    for name in ("target_mae", "target_rmse", "target_bias", "target_pearson"):
        short_name = name.removeprefix("target_")
        metrics[f"{prefix}/baseline_vs_reward/{short_name}"] = (
            baseline_metrics[name]
        )

    advantage_identity_errors = arrays["advantage"] - (
        reward_values - baseline_values
    )
    metrics[f"{prefix}/advantage_identity/mean_abs_error"] = float(
        np.mean(np.abs(advantage_identity_errors))
    )
    metrics[f"{prefix}/advantage_identity/max_abs_error"] = float(
        np.max(np.abs(advantage_identity_errors))
    )
    metrics[f"{prefix}/target_reward_mean_gap"] = float(
        arrays["estimator_target"].mean() - reward_values.mean()
    )
    metrics[f"{prefix}/baseline_prediction_mean_gap"] = float(
        baseline_values.mean() - arrays["value_prediction"].mean()
    )
    return metrics
