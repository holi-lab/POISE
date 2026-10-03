"""Domain-routed estimator state for multi-domain POISE."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .estimator import (
    EstimatorFitConfig,
    FeatureBuilderConfig,
    SingleTrajectoryEstimator,
    SingleTrajectoryFeatureBuilder,
    fit_estimator,
    load_estimator,
)


DOMAIN_NAMES = ("math", "code", "other")
STATE_FILENAME = "poise_domain_estimator_state.pt"
STATE_SCHEMA_VERSION = 2


def domain_from_data_source(data_source: object) -> str:
    """Map GURU data sources to the math, code, and other probe domains."""
    source = str(data_source).strip().lower()
    if source.startswith("math"):
        return "math"
    if source.startswith("codegen"):
        return "code"
    return "other"


@dataclass
class DomainEstimatorState:
    estimator: SingleTrajectoryEstimator
    buffer_steps: list[list[dict[str, Any]]] = field(default_factory=list)
    observed_steps: int = 0
    retrain_count: int = 0

    def flattened_rows(self) -> list[dict[str, Any]]:
        return [row for step_rows in self.buffer_steps for row in step_rows]


class RecentDomainBuffer:
    """Keep the newest complete prompt groups under per-domain row limits."""

    def __init__(self, buffer_max_rows: dict[str, int]) -> None:
        self.buffer_max_rows = {
            domain: int(buffer_max_rows[domain]) for domain in DOMAIN_NAMES
        }
        self.groups: dict[str, list[list[dict[str, Any]]]] = {
            domain: [] for domain in DOMAIN_NAMES
        }
        self.row_counts = {domain: 0 for domain in DOMAIN_NAMES}

    def append(self, rows_by_domain: dict[str, list[dict[str, Any]]]) -> None:
        for domain in DOMAIN_NAMES:
            uid_groups: dict[str, list[dict[str, Any]]] = {}
            for row in rows_by_domain.get(domain, ()):
                uid_groups.setdefault(str(row["uid"]), []).append(row)
            for rows in uid_groups.values():
                self.groups[domain].append(rows)
                self.row_counts[domain] += len(rows)
            while (
                self.groups[domain]
                and self.row_counts[domain] > self.buffer_max_rows[domain]
            ):
                removed = self.groups[domain].pop(0)
                self.row_counts[domain] -= len(removed)

    def rows_by_domain(self) -> dict[str, list[dict[str, Any]]]:
        return {
            domain: [
                row
                for group_rows in self.groups[domain]
                for row in group_rows
            ]
            for domain in DOMAIN_NAMES
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "buffer_max_rows": dict(self.buffer_max_rows),
            "groups": {
                domain: list(self.groups[domain]) for domain in DOMAIN_NAMES
            },
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        expected_buffer_max_rows: dict[str, int],
    ) -> "RecentDomainBuffer":
        buffer = cls(expected_buffer_max_rows)
        saved_limits = {
            domain: int(payload["buffer_max_rows"][domain])
            for domain in DOMAIN_NAMES
        }
        if saved_limits != buffer.buffer_max_rows:
            raise ValueError(
                "POISE bootstrap buffer limits changed across resume: "
                f"{saved_limits} != {buffer.buffer_max_rows}"
            )
        buffer.groups = {
            domain: [list(group) for group in payload["groups"][domain]]
            for domain in DOMAIN_NAMES
        }
        buffer.row_counts = {
            domain: sum(len(group) for group in buffer.groups[domain])
            for domain in DOMAIN_NAMES
        }
        for domain in DOMAIN_NAMES:
            if buffer.row_counts[domain] > buffer.buffer_max_rows[domain]:
                raise ValueError(
                    f"POISE bootstrap {domain} buffer exceeds its row limit"
                )
        return buffer


class DomainEstimatorBank:
    """Own one independently initialized and independently refitted estimator per domain."""

    def __init__(
        self,
        *,
        model_paths: dict[str, str],
        feature_config_path: str,
        fit_config_path: str,
        warmup_steps: int,
        buffer_max_rows: dict[str, int],
    ) -> None:
        missing = sorted(set(DOMAIN_NAMES) - set(model_paths))
        if missing:
            raise ValueError(f"Missing POISE initial estimator paths for domains: {missing}")
        missing_buffer_limits = sorted(set(DOMAIN_NAMES) - set(buffer_max_rows))
        if missing_buffer_limits:
            raise ValueError(
                "Missing POISE buffer row limits for domains: "
                f"{missing_buffer_limits}"
            )
        if warmup_steps <= 0:
            raise ValueError("POISE estimator warmup_steps must be positive")
        invalid_buffer_limits = {
            domain: int(buffer_max_rows[domain])
            for domain in DOMAIN_NAMES
            if int(buffer_max_rows[domain]) <= 0
        }
        if invalid_buffer_limits:
            raise ValueError(
                "POISE estimator buffer_max_rows must be positive: "
                f"{invalid_buffer_limits}"
            )

        self._initialize_runtime(
            feature_config_path=feature_config_path,
            fit_config_path=fit_config_path,
            warmup_steps=warmup_steps,
            buffer_max_rows=buffer_max_rows,
        )
        self.initial_model_paths = {
            domain: str(Path(model_paths[domain]).expanduser().resolve())
            for domain in DOMAIN_NAMES
        }
        self.states = {
            domain: DomainEstimatorState(load_estimator(self.initial_model_paths[domain]))
            for domain in DOMAIN_NAMES
        }
        for domain, state in self.states.items():
            self._validate_estimator_feature_keys(domain, state.estimator)

    def _initialize_runtime(
        self,
        *,
        feature_config_path: str,
        fit_config_path: str,
        warmup_steps: int,
        buffer_max_rows: dict[str, int],
    ) -> None:
        missing_buffer_limits = sorted(set(DOMAIN_NAMES) - set(buffer_max_rows))
        if missing_buffer_limits:
            raise ValueError(
                "Missing POISE buffer row limits for domains: "
                f"{missing_buffer_limits}"
            )
        if int(warmup_steps) <= 0:
            raise ValueError("POISE estimator warmup_steps must be positive")
        invalid_buffer_limits = {
            domain: int(buffer_max_rows[domain])
            for domain in DOMAIN_NAMES
            if int(buffer_max_rows[domain]) <= 0
        }
        if invalid_buffer_limits:
            raise ValueError(
                "POISE estimator buffer_max_rows must be positive: "
                f"{invalid_buffer_limits}"
            )
        feature_path = Path(feature_config_path).expanduser().resolve()
        fit_path = Path(fit_config_path).expanduser().resolve()
        with feature_path.open(encoding="utf-8") as handle:
            self.feature_config = FeatureBuilderConfig.from_dict(json.load(handle))
        with fit_path.open(encoding="utf-8") as handle:
            self.fit_config = EstimatorFitConfig.from_dict(json.load(handle))
        self.feature_builder = SingleTrajectoryFeatureBuilder(self.feature_config)
        self.warmup_steps = int(warmup_steps)
        self.buffer_max_rows = {
            domain: int(buffer_max_rows[domain]) for domain in DOMAIN_NAMES
        }

    @classmethod
    def runtime(
        cls,
        *,
        feature_config_path: str,
        fit_config_path: str,
        warmup_steps: int,
        buffer_max_rows: dict[str, int],
    ) -> "DomainEstimatorBank":
        """Load feature/fit configuration without requiring estimators."""
        bank = cls.__new__(cls)
        bank._initialize_runtime(
            feature_config_path=feature_config_path,
            fit_config_path=fit_config_path,
            warmup_steps=warmup_steps,
            buffer_max_rows=buffer_max_rows,
        )
        bank.initial_model_paths = {}
        bank.states = {}
        return bank

    @classmethod
    def from_rows(
        cls,
        *,
        rows_by_domain: dict[str, list[dict[str, Any]]],
        feature_config_path: str,
        fit_config_path: str,
        warmup_steps: int,
        buffer_max_rows: dict[str, int],
        observed_steps: int,
    ) -> tuple["DomainEstimatorBank", dict[str, float]]:
        bank = cls.__new__(cls)
        bank._initialize_runtime(
            feature_config_path=feature_config_path,
            fit_config_path=fit_config_path,
            warmup_steps=warmup_steps,
            buffer_max_rows=buffer_max_rows,
        )
        bank.initial_model_paths = {}
        bank.states = {}
        metrics: dict[str, float] = {}
        for domain in DOMAIN_NAMES:
            rows = list(rows_by_domain.get(domain, ()))
            if not rows:
                raise ValueError(
                    f"Cannot bootstrap the POISE {domain} estimator without rows"
                )
            estimator, fit_metrics = fit_estimator(
                rows=rows,
                feature_config=bank.feature_config,
                fit_config=bank.fit_config,
            )
            bank._validate_estimator_feature_keys(domain, estimator)
            bank.states[domain] = DomainEstimatorState(
                estimator=estimator,
                buffer_steps=bank._rows_grouped_by_global_step(rows),
                observed_steps=int(observed_steps),
                retrain_count=1,
            )
            prefix = f"poise/estimator/{domain}"
            metrics[f"{prefix}/bootstrap_rows"] = float(len(rows))
            metrics[f"{prefix}/retrain_count"] = 1.0
            metrics.update(
                {
                    f"{prefix}/bootstrap_{name}": value
                    for name, value in fit_metrics.items()
                }
            )
        return bank, metrics

    @staticmethod
    def _rows_grouped_by_global_step(
        rows: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        if not rows or any(row.get("global_step") is None for row in rows):
            return [rows] if rows else []
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["global_step"]), []).append(row)
        return [grouped[step] for step in sorted(grouped)]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | os.PathLike[str],
        *,
        feature_config_path: str,
        fit_config_path: str,
        warmup_steps: int,
        buffer_max_rows: dict[str, int],
    ) -> "DomainEstimatorBank":
        bank = cls.__new__(cls)
        bank._initialize_runtime(
            feature_config_path=feature_config_path,
            fit_config_path=fit_config_path,
            warmup_steps=warmup_steps,
            buffer_max_rows=buffer_max_rows,
        )
        bank.initial_model_paths = {}
        bank.states = {}
        if not bank.load(checkpoint_dir):
            raise FileNotFoundError(
                "POISE estimator checkpoint is missing: "
                f"{Path(checkpoint_dir) / STATE_FILENAME}"
            )
        return bank

    def _configured_feature_keys(self) -> tuple[str, ...]:
        configured_feature_keys = tuple(self.feature_config.rollout_scalars.scalar_keys)
        configured_feature_keys += tuple(
            self.feature_config.rollout_scalars.derived_scalar_keys
        )
        configured_feature_keys += tuple(
            path.replace(".", "_")
            for path in self.feature_config.rollout_scalars.extra_scalar_field_paths
        )
        return configured_feature_keys

    def _validate_estimator_feature_keys(
        self,
        domain: str,
        estimator: SingleTrajectoryEstimator,
    ) -> None:
        estimator_feature_keys = tuple(
            estimator.config.response_feature_keys
        ) + tuple(estimator.config.derived_response_feature_keys)
        configured_feature_keys = self._configured_feature_keys()
        if estimator_feature_keys != configured_feature_keys:
            raise ValueError(
                f"POISE {domain} estimator features do not match the feature "
                f"builder config: {estimator_feature_keys} != "
                f"{configured_feature_keys}"
            )

    @property
    def capture_spec(self) -> dict[str, Any]:
        prompt = self.feature_config.prompt_hidden
        response = self.feature_config.response_hidden
        if prompt.pooling.type != "last_n_mean" or response.pooling.type != "last_n_mean":
            raise ValueError("POISE hidden capture requires last_n_mean pooling")
        if prompt.pooling.n <= 0 or response.pooling.n <= 0:
            raise ValueError("POISE last_n_mean pooling requires positive n values")
        spec = {
            "prompt_layer_index": int(prompt.layer_index),
            "response_layer_index": int(response.layer_index),
            "prompt_pool_n": int(prompt.pooling.n),
            "response_pool_n": int(response.pooling.n),
        }
        # Keep the legacy field for equal-layer configurations so older worker
        # implementations can still consume checkpoints/configs produced by
        # the generalized capture path.
        if prompt.layer_index == response.layer_index:
            spec["layer_index"] = int(prompt.layer_index)
        return spec

    def estimator_for(self, domain: str) -> SingleTrajectoryEstimator:
        if domain not in self.states:
            raise KeyError(f"Unknown POISE domain: {domain}")
        return self.states[domain].estimator

    @staticmethod
    def _uid_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups: dict[tuple[str, str | int], list[dict[str, Any]]] = {}
        for index, row in enumerate(rows):
            uid = row.get("uid")
            key = ("uid", str(uid)) if uid is not None else ("row", index)
            groups.setdefault(key, []).append(row)
        return list(groups.values())

    def _trim_buffer_rows(self, domain: str) -> int:
        """Drop the oldest recent prompt groups until the row limit holds."""
        state = self.states[domain]
        max_rows = self.buffer_max_rows[domain]
        row_count = sum(len(step_rows) for step_rows in state.buffer_steps)
        while state.buffer_steps and row_count > max_rows:
            oldest_groups = self._uid_groups(state.buffer_steps[0])
            overflow = row_count - max_rows
            removed_rows = 0
            while oldest_groups and removed_rows < overflow:
                removed_rows += len(oldest_groups.pop(0))
            if oldest_groups:
                state.buffer_steps[0] = [
                    row for group_rows in oldest_groups for row in group_rows
                ]
            else:
                state.buffer_steps.pop(0)
            row_count -= removed_rows
        return row_count

    def update(
        self, rows_by_domain: dict[str, list[dict[str, Any]]]
    ) -> dict[str, float]:
        """Append one actor update of rows and refit eligible domain estimators."""
        metrics: dict[str, float] = {}
        for domain in DOMAIN_NAMES:
            state = self.states[domain]
            rows = list(rows_by_domain.get(domain, ()))
            prefix = f"poise/estimator/{domain}"
            metrics[f"{prefix}/rows_added"] = float(len(rows))
            if not rows:
                recent_rows = state.flattened_rows()
                metrics[f"{prefix}/refit"] = 0.0
                metrics[f"{prefix}/buffer_rows"] = float(len(recent_rows))
                metrics[f"{prefix}/buffer_max_rows"] = float(
                    self.buffer_max_rows[domain]
                )
                metrics[f"{prefix}/recent_buffer_rows"] = float(len(recent_rows))
                metrics[f"{prefix}/fit_rows"] = float(len(recent_rows))
                continue

            state.buffer_steps.append(rows)
            state.observed_steps += 1
            self._trim_buffer_rows(domain)
            buffered_rows = state.flattened_rows()
            metrics[f"{prefix}/observed_steps"] = float(state.observed_steps)
            metrics[f"{prefix}/buffer_steps"] = float(len(state.buffer_steps))
            # Keep buffer_rows as the historical recent-FIFO metric for
            # dashboard compatibility; fit_rows is the actual refit size.
            metrics[f"{prefix}/buffer_rows"] = float(len(buffered_rows))
            metrics[f"{prefix}/buffer_max_rows"] = float(
                self.buffer_max_rows[domain]
            )
            metrics[f"{prefix}/recent_buffer_rows"] = float(len(buffered_rows))
            metrics[f"{prefix}/fit_rows"] = float(len(buffered_rows))
            metrics[f"{prefix}/warmup_remaining"] = float(
                max(0, self.warmup_steps - state.observed_steps)
            )

            if state.observed_steps < self.warmup_steps:
                metrics[f"{prefix}/refit"] = 0.0
                continue

            state.estimator, fit_metrics = fit_estimator(
                rows=buffered_rows,
                feature_config=self.feature_config,
                fit_config=self.fit_config,
            )
            state.retrain_count += 1
            metrics[f"{prefix}/refit"] = 1.0
            metrics[f"{prefix}/retrain_count"] = float(state.retrain_count)
            metrics.update(
                {
                    f"{prefix}/{metric_name}": metric_value
                    for metric_name, metric_value in fit_metrics.items()
                }
            )
        return metrics

    def adaptive_training_data_payload(self, *, global_step: int) -> dict[str, Any]:
        """Export the exact recent-buffer refit rows in flat fields."""
        step_buckets: dict[
            tuple[Any, ...], list[tuple[str, str, dict[str, Any]]]
        ] = {}
        domain_summaries: dict[str, dict[str, int]] = {}

        def add_step_rows(
            *,
            source: str,
            domain: str,
            position: int,
            rows: list[dict[str, Any]],
        ) -> None:
            explicit_steps = {
                int(row["global_step"])
                for row in rows
                if row.get("global_step") is not None
            }
            if len(explicit_steps) == 1 and all(
                row.get("global_step") is not None for row in rows
            ):
                bucket_key: tuple[Any, ...] = ("global", explicit_steps.pop())
            else:
                # Compatibility fallback for checkpoints created before rows
                # carried their source global step.
                bucket_key = ("legacy", source, domain, position)
            step_buckets.setdefault(bucket_key, []).extend(
                (source, domain, row) for row in rows
            )

        for domain in DOMAIN_NAMES:
            state = self.states[domain]
            recent_rows = state.flattened_rows()
            domain_summaries[domain] = {
                "observed_steps": int(state.observed_steps),
                "retrain_count": int(state.retrain_count),
                "buffer_steps": int(len(state.buffer_steps)),
                "buffer_rows": int(len(recent_rows)),
                "recent_buffer_rows": int(len(recent_rows)),
                "fit_rows": int(len(recent_rows)),
            }
            for step_position, step_rows in enumerate(state.buffer_steps):
                add_step_rows(
                    source="recent",
                    domain=domain,
                    position=step_position,
                    rows=step_rows,
                )

        global_keys = sorted(
            (key for key in step_buckets if key[0] == "global"),
            key=lambda key: int(key[1]),
        )
        legacy_keys = [key for key in step_buckets if key[0] == "legacy"]
        ordered_keys = [*legacy_keys, *global_keys]
        ordered_rows = [
            (bucket_key, source, domain, row)
            for bucket_key in ordered_keys
            for source, domain, row in step_buckets[bucket_key]
        ]

        required_row_keys = {
            "prompt_hidden",
            "response_hidden",
            "response_features",
            "target",
            "uid",
        }
        for _, _, domain, row in ordered_rows:
            missing = required_row_keys - set(row)
            if missing:
                raise KeyError(
                    f"POISE {domain} estimator buffer row is missing snapshot "
                    f"fields: {sorted(missing)}"
                )

        group_rewards: dict[str, list[float]] = {}
        group_keys: list[str] = []
        for bucket_key, _, domain, row in ordered_rows:
            step_label = (
                str(bucket_key[1])
                if bucket_key[0] == "global"
                else ":".join(str(value) for value in bucket_key)
            )
            group_key = f"{step_label}:{domain}:{row['uid']}"
            group_keys.append(group_key)
            group_rewards.setdefault(group_key, []).append(
                float(row.get("reward", row["target"]))
            )

        def group_type(rewards: list[float], epsilon: float = 1e-8) -> str:
            if rewards and all(abs(reward) <= epsilon for reward in rewards):
                return "all0"
            if rewards and all(abs(reward - 1.0) <= epsilon for reward in rewards):
                return "all1"
            return "mixed"

        estimator_files = {
            domain: f"poise_{domain}_estimator.joblib"
            for domain in DOMAIN_NAMES
        }
        refit_domains = [
            domain
            for domain in DOMAIN_NAMES
            if self.states[domain].buffer_steps
            and any(
                int(row.get("global_step", -1)) == int(global_step)
                for row in self.states[domain].buffer_steps[-1]
            )
            and self.states[domain].observed_steps >= self.warmup_steps
        ]
        return {
            "schema_version": 3,
            "global_step": int(global_step),
            "retrain_steps": max(
                (state.observed_steps for state in self.states.values()),
                default=0,
            ),
            "retrain_count": max(
                (state.retrain_count for state in self.states.values()),
                default=0,
            ),
            "retrain_steps_by_domain": {
                domain: int(self.states[domain].observed_steps)
                for domain in DOMAIN_NAMES
            },
            "retrain_count_by_domain": {
                domain: int(self.states[domain].retrain_count)
                for domain in DOMAIN_NAMES
            },
            "refit_domains_this_step": refit_domains,
            "estimator_trained_from_saved_data": bool(refit_domains),
            "estimator_files": estimator_files,
            "buffer_rows": len(ordered_rows),
            "recent_buffer_rows": sum(
                len(state.flattened_rows()) for state in self.states.values()
            ),
            "buffer_steps": len(ordered_keys),
            "buffer_step_row_counts": [
                len(step_buckets[key]) for key in ordered_keys
            ],
            "buffer_max_rows_by_domain": dict(self.buffer_max_rows),
            "domain_summaries": domain_summaries,
            "train_buffer_sources": [
                source for _, source, _, _ in ordered_rows
            ],
            "train_domains": [domain for _, _, domain, _ in ordered_rows],
            "train_prompt_hidden_rows": [
                np.asarray(row["prompt_hidden"], dtype=np.float32).reshape(-1)
                for _, _, _, row in ordered_rows
            ],
            "train_response_hidden_rows": [
                np.asarray(row["response_hidden"], dtype=np.float32).reshape(-1)
                for _, _, _, row in ordered_rows
            ],
            "train_response_feature_rows": [
                {
                    str(key): float(value)
                    for key, value in row["response_features"].items()
                }
                for _, _, _, row in ordered_rows
            ],
            "train_targets": [
                float(row["target"]) for _, _, _, row in ordered_rows
            ],
            "train_rewards": [
                float(row.get("reward", row["target"]))
                for _, _, _, row in ordered_rows
            ],
            "train_group_keys": group_keys,
            "train_group_types": [
                group_type(group_rewards[key]) for key in group_keys
            ],
        }

    def save(self, checkpoint_dir: str | os.PathLike[str]) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("POISE requires joblib to save estimators") from exc

        output_dir = Path(checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_suffix = f".tmp-{os.getpid()}"
        estimator_files: dict[str, str] = {}
        temporary_estimator_paths: dict[str, Path] = {}
        temporary_state_path = output_dir / f".{STATE_FILENAME}{temporary_suffix}"
        try:
            for domain in DOMAIN_NAMES:
                filename = f"poise_{domain}_estimator.joblib"
                temporary_path = output_dir / f".{filename}{temporary_suffix}"
                joblib.dump(
                    self.states[domain].estimator.to_bundle(),
                    temporary_path,
                )
                estimator_files[domain] = filename
                temporary_estimator_paths[domain] = temporary_path
            payload = {
                "schema_version": STATE_SCHEMA_VERSION,
                "warmup_steps": self.warmup_steps,
                "buffer_max_rows": dict(self.buffer_max_rows),
                "estimator_files": estimator_files,
                "domains": {
                    domain: {
                        "buffer_steps": self.states[domain].buffer_steps,
                        "observed_steps": self.states[domain].observed_steps,
                        "retrain_count": self.states[domain].retrain_count,
                    }
                    for domain in DOMAIN_NAMES
                },
            }
            torch.save(payload, temporary_state_path)
            for domain, temporary_path in temporary_estimator_paths.items():
                os.replace(
                    temporary_path,
                    output_dir / estimator_files[domain],
                )
            # Commit the state file last: its presence means all referenced
            # estimator bundles have already been committed.
            os.replace(temporary_state_path, output_dir / STATE_FILENAME)
        finally:
            for temporary_path in temporary_estimator_paths.values():
                temporary_path.unlink(missing_ok=True)
            temporary_state_path.unlink(missing_ok=True)

    def load(self, checkpoint_dir: str | os.PathLike[str]) -> bool:
        checkpoint_path = Path(checkpoint_dir)
        state_path = checkpoint_path / STATE_FILENAME
        if not state_path.exists():
            return False
        payload = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid POISE estimator state: {state_path}")
        schema_version = int(payload.get("schema_version", 0))
        if schema_version > STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported POISE estimator state schema {schema_version}; "
                f"runtime supports up to {STATE_SCHEMA_VERSION}."
            )
        if (
            "warmup_steps" in payload
            and int(payload["warmup_steps"]) != self.warmup_steps
        ):
            raise ValueError(
                "POISE estimator warmup_steps changed across resume: "
                f"{payload['warmup_steps']} != {self.warmup_steps}"
            )
        if "buffer_max_rows" in payload:
            saved_buffer_limits = {
                domain: int(payload["buffer_max_rows"][domain])
                for domain in DOMAIN_NAMES
            }
            if saved_buffer_limits != self.buffer_max_rows:
                raise ValueError(
                    "POISE estimator buffer_max_rows changed across resume: "
                    f"{saved_buffer_limits} != {self.buffer_max_rows}"
                )
        estimator_files = payload.get("estimator_files")
        domain_payloads = payload.get("domains")
        if not isinstance(estimator_files, dict) or not isinstance(
            domain_payloads, dict
        ):
            raise ValueError(
                f"POISE estimator state is missing domain metadata: {state_path}"
            )

        loaded_states: dict[str, DomainEstimatorState] = {}
        for domain in DOMAIN_NAMES:
            if domain not in estimator_files or domain not in domain_payloads:
                raise ValueError(
                    f"POISE estimator state is missing domain {domain!r}: "
                    f"{state_path}"
                )
            estimator_path = checkpoint_path / estimator_files[domain]
            if not estimator_path.is_file():
                raise FileNotFoundError(
                    f"POISE {domain} estimator bundle is missing: "
                    f"{estimator_path}"
                )
            estimator = load_estimator(estimator_path)
            self._validate_estimator_feature_keys(domain, estimator)
            domain_payload = domain_payloads[domain]
            observed_steps = int(domain_payload["observed_steps"])
            retrain_count = int(domain_payload["retrain_count"])
            if observed_steps < 0 or retrain_count < 0:
                raise ValueError(
                    f"POISE {domain} estimator counters cannot be negative"
                )
            loaded_states[domain] = DomainEstimatorState(
                estimator=estimator,
                buffer_steps=[
                    list(step_rows)
                    for step_rows in domain_payload["buffer_steps"]
                ],
                observed_steps=observed_steps,
                retrain_count=retrain_count,
            )

        # Do not partially mutate a running bank if one domain failed validation.
        self.states = loaded_states
        for domain in DOMAIN_NAMES:
            self._trim_buffer_rows(domain)
        return True

    def domain_counts(self, domains: Iterable[str]) -> dict[str, int]:
        counts = {domain: 0 for domain in DOMAIN_NAMES}
        for domain in domains:
            counts[domain] += 1
        return counts
