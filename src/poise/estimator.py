"""Runtime and online fitting support for POISE single-trajectory estimators.

Estimator bundles contain a fitted regressor, optional prompt/response PCA
objects, and a serializable config payload. Historical initial bundles are
supported through explicit serialization aliases.
"""

from __future__ import annotations

import math
import re
import sys
import types
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .core import regression_diagnostic_values


REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS = frozenset(
    {
        "output_mean_token_entropy",
        "output_last_token_entropy",
        "output_max_token_entropy",
        "output_min_token_entropy",
        "reasoning_mean_token_entropy",
        "reasoning_last_token_entropy",
        "reasoning_max_token_entropy",
        "reasoning_min_token_entropy",
        "answer_mean_token_entropy",
        "answer_last_token_entropy",
        "answer_max_token_entropy",
        "answer_min_token_entropy",
    }
)
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.I | re.S)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _nested_value(record: dict[str, Any], field_path: str) -> Any:
    value: Any = record
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _sanitize_name(field_path: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", field_path).strip("_")


def _tokenize_whitespace(text: str) -> list[str]:
    return re.findall(r"\S+", text.lower())


def _unique_token_ratio(text: str) -> float | None:
    tokens = _tokenize_whitespace(text)
    return len(set(tokens)) / len(tokens) if tokens else None


def _repetition_ratio(text: str) -> float | None:
    unique_ratio = _unique_token_ratio(text)
    return None if unique_ratio is None else 1.0 - unique_ratio


def _duplicate_line_ratio(text: str) -> float | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return None if not lines else 1.0 - len(set(lines)) / len(lines)


def _reasoning_and_answer(generated_text: str) -> tuple[str, str]:
    think_match = _THINK_PATTERN.search(generated_text)
    answer_match = _ANSWER_PATTERN.search(generated_text)
    reasoning = think_match.group(1) if think_match else ""
    if answer_match:
        answer = answer_match.group(1)
    elif "</think>" in generated_text:
        answer = generated_text.split("</think>", maxsplit=1)[1].strip()
    else:
        answer = ""
    return reasoning, answer


def _count_text_tokens(tokenizer: Any | None, text: str) -> int:
    if tokenizer is None or not text.strip():
        return 0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


class FastPCA:
    """Compatibility shim for PCA objects serialized by the poise recipe."""

    def transform(
        self, values: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError(f"Expected a rank-2 PCA input, got {matrix.shape}")
        components = np.asarray(self.components_, dtype=np.float32)
        mean = np.asarray(
            getattr(self, "mean_", np.zeros(components.shape[1])),
            dtype=np.float32,
        )
        return ((matrix - mean.reshape(1, -1)) @ components.T).astype(
            np.float32, copy=False
        )


class SigmoidLinearRegressor:
    """Compatibility runtime for poise sigmoid-BCE estimator bundles."""

    def __init__(
        self,
        *,
        mean: np.ndarray,
        scale: np.ndarray,
        coef: np.ndarray,
        intercept: float,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.scale_ = np.asarray(scale, dtype=np.float32).reshape(1, -1)
        self.coef_ = np.asarray(coef, dtype=np.float32).reshape(-1)
        self.intercept_ = float(intercept)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.alpha = None
        self.n_features_in_ = int(self.coef_.shape[0])

    def predict(self, values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        logits = (
            (matrix - np.asarray(self.mean_, dtype=np.float32))
            / np.asarray(self.scale_, dtype=np.float32)
        ) @ np.asarray(self.coef_, dtype=np.float32).reshape(-1, 1)
        logits = logits.reshape(-1) + float(self.intercept_)
        predictions = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        return np.clip(
            predictions,
            float(getattr(self, "clip_min", 0.0)),
            float(getattr(self, "clip_max", 1.0)),
        ).astype(np.float32, copy=False)


class MultiEstimatorRegressor:
    """Compatibility runtime for poise multi-estimator bundles."""

    def __init__(
        self,
        *,
        estimators: Sequence[Any],
        combine_method: str,
        stacker: Any | None,
        clip_min: float,
        clip_max: float,
        alpha: float,
    ) -> None:
        if not estimators:
            raise ValueError("MultiEstimatorRegressor requires child estimators")
        if combine_method not in {"mean", "ridge_stacker"}:
            raise ValueError(f"Unsupported multi-estimator combine: {combine_method}")
        if combine_method == "ridge_stacker" and stacker is None:
            raise ValueError("ridge_stacker requires a fitted stacker")
        self.estimators_ = list(estimators)
        self.combine_method = combine_method
        self.stacker_ = stacker
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.alpha = float(alpha)
        self.n_features_in_ = int(
            getattr(self.estimators_[0], "n_features_in_", 0) or 0
        )

    def _child_prediction_matrix(
        self, values: np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        predictions = [
            np.asarray(estimator.predict(matrix), dtype=np.float32).reshape(-1)
            for estimator in self.estimators_
        ]
        return np.stack(predictions, axis=1).astype(np.float32, copy=False)

    def predict(self, values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        child_predictions = self._child_prediction_matrix(values)
        if self.combine_method == "mean":
            predictions = child_predictions.mean(axis=1)
        else:
            predictions = np.asarray(
                self.stacker_.predict(child_predictions), dtype=np.float32
            ).reshape(-1)
        return np.clip(
            predictions,
            float(getattr(self, "clip_min", 0.0)),
            float(getattr(self, "clip_max", 1.0)),
        ).astype(np.float32, copy=False)

    def predict_children(
        self, values: np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        return self._child_prediction_matrix(values)


def _register_legacy_pickle_aliases() -> None:
    """Make joblibs written from poise importable without installing poise."""

    import __main__

    for class_name, class_object in (
        ("FastPCA", FastPCA),
        ("SigmoidLinearRegressor", SigmoidLinearRegressor),
        ("MultiEstimatorRegressor", MultiEstimatorRegressor),
    ):
        if not hasattr(__main__, class_name):
            setattr(__main__, class_name, class_object)

    module_names = (
        "recipe.CrossRolloutRL.estimator.single_trajectory_estimator_support."
        "value_estimator.runtime",
        "classifer_training.sweep_spo_base_rowr2_axis",
    )
    for module_name in module_names:
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            current_name = ".".join(parts[:depth])
            module = sys.modules.get(current_name)
            if module is None:
                module = types.ModuleType(current_name)
                if depth < len(parts):
                    module.__path__ = []
                sys.modules[current_name] = module
            if depth > 1:
                parent_name = ".".join(parts[: depth - 1])
                setattr(sys.modules[parent_name], parts[depth - 1], module)
        runtime_module = sys.modules[module_name]
        runtime_module.FastPCA = FastPCA
        runtime_module.SigmoidLinearRegressor = SigmoidLinearRegressor
        runtime_module.MultiEstimatorRegressor = MultiEstimatorRegressor


@dataclass(frozen=True)
class PoolingConfig:
    type: str
    n: int


@dataclass(frozen=True)
class HiddenSequenceConfig:
    input_field: str
    layer_index: int
    pooling: PoolingConfig


@dataclass(frozen=True)
class RolloutScalarConfig:
    scalar_keys: tuple[str, ...]
    derived_scalar_keys: tuple[str, ...] = ()
    extra_scalar_field_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureBuilderConfig:
    prompt_hidden: HiddenSequenceConfig
    response_hidden: HiddenSequenceConfig
    rollout_scalars: RolloutScalarConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureBuilderConfig":
        prompt = dict(payload["prompt_hidden"])
        response = dict(
            payload.get(
                "response_hidden",
                payload.get("think_end_hidden", payload.get("trajectory_hidden")),
            )
        )
        rollout = dict(payload["rollout_scalars"])
        return cls(
            prompt_hidden=HiddenSequenceConfig(
                input_field=str(prompt["input_field"]),
                layer_index=int(prompt["layer_index"]),
                pooling=PoolingConfig(
                    type=str(prompt["pooling"]["type"]),
                    n=int(prompt["pooling"]["n"]),
                ),
            ),
            response_hidden=HiddenSequenceConfig(
                input_field=str(response["input_field"]),
                layer_index=int(response["layer_index"]),
                pooling=PoolingConfig(
                    type=str(response["pooling"]["type"]),
                    n=int(response["pooling"]["n"]),
                ),
            ),
            rollout_scalars=RolloutScalarConfig(
                scalar_keys=tuple(rollout.get("scalar_keys", ())),
                derived_scalar_keys=tuple(rollout.get("derived_scalar_keys", ())),
                extra_scalar_field_paths=tuple(
                    rollout.get("extra_scalar_field_paths", ())
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EstimatorFitConfig:
    prompt_hidden_pca_dim: int = 0
    response_hidden_pca_dim: int = 0
    alpha: float = 0.01
    model_type: str = "ridge"
    child_fit_configs: tuple[dict[str, Any], ...] | list[dict[str, Any]] = ()
    multi_estimator_combine: str = "mean"
    multi_estimator_stacker_alpha: float = 0.01
    target_mode: str = "other_rollout_correctness"
    random_seed: int = 42
    clip_min: float = 0.0
    clip_max: float = 1.0
    sigmoid_bce_max_iter: int = 500
    sigmoid_bce_eps: float = 1e-6
    mlp_hidden_dim: int = 64
    mlp_epochs: int = 1000
    mlp_lr: float = 1e-3
    mlp_batch_size: int = 8192

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EstimatorFitConfig":
        supported = set(cls.__dataclass_fields__)
        unknown = set(payload) - supported
        if unknown:
            raise ValueError(
                f"Unsupported estimator fit config keys: {sorted(unknown)}"
            )
        config = cls(**payload)
        config = replace(
            config,
            model_type=str(config.model_type).strip().lower(),
            multi_estimator_combine=str(
                config.multi_estimator_combine
            ).strip().lower(),
        )
        if config.model_type not in {"ridge", "sigmoid_bce", "mlp"}:
            raise ValueError(
                "POISE model_type must be one of: ridge, sigmoid_bce, mlp"
            )
        if config.multi_estimator_combine not in {"mean", "ridge_stacker"}:
            raise ValueError(
                "multi_estimator_combine must be 'mean' or 'ridge_stacker'"
            )
        return config


@dataclass(frozen=True)
class ProjectionConfig:
    type: str | None
    input_dim: int | None
    output_dim: int | None


@dataclass(frozen=True)
class EstimatorModelConfig:
    alpha: float
    clip_min: float
    clip_max: float
    feature_dim: int


@dataclass(frozen=True)
class EstimatorConfig:
    prompt_hidden_projection: ProjectionConfig
    response_hidden_projection: ProjectionConfig
    response_feature_keys: tuple[str, ...]
    derived_response_feature_keys: tuple[str, ...]
    model: EstimatorModelConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EstimatorConfig":
        return cls(
            prompt_hidden_projection=ProjectionConfig(
                **dict(payload["prompt_hidden_projection"])
            ),
            response_hidden_projection=ProjectionConfig(
                **dict(
                    payload.get(
                        "response_hidden_projection",
                        payload.get(
                            "think_end_hidden_projection",
                            payload.get("trajectory_hidden_projection"),
                        ),
                    )
                )
            ),
            response_feature_keys=tuple(
                payload.get(
                    "response_feature_keys",
                    payload.get("trajectory_scalar_keys", ()),
                )
            ),
            derived_response_feature_keys=tuple(
                payload.get(
                    "derived_response_feature_keys",
                    payload.get("trajectory_derived_scalar_keys", ()),
                )
            ),
            model=EstimatorModelConfig(**dict(payload["model"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _projection_config(projection: Any | None) -> ProjectionConfig:
    if projection is None:
        return ProjectionConfig(type=None, input_dim=None, output_dim=None)
    components = np.asarray(projection.components_)
    return ProjectionConfig(
        type="pca",
        input_dim=int(getattr(projection, "n_features_in_", components.shape[1])),
        output_dim=int(getattr(projection, "n_components_", components.shape[0])),
    )


def _estimator_alpha(estimator: Any) -> float:
    named_steps = getattr(estimator, "named_steps", {})
    model = named_steps.get("model", estimator)
    return float(getattr(model, "alpha", 0.01))


def _legacy_config(
    payload: dict[str, Any],
    *,
    estimator: Any,
    prompt_projection: Any | None,
    response_projection: Any | None,
) -> EstimatorConfig:
    model_description = payload.get("model")
    clip_min, clip_max = 0.0, 1.0
    if isinstance(model_description, str):
        match = re.search(r"clip\[\s*([^,]+),\s*([^\]]+)\]", model_description)
        if match:
            clip_min, clip_max = float(match.group(1)), float(match.group(2))
    feature_dim = int(
        payload.get("feature_dim", getattr(estimator, "n_features_in_", 0) or 0)
    )
    return EstimatorConfig(
        prompt_hidden_projection=_projection_config(prompt_projection),
        response_hidden_projection=_projection_config(response_projection),
        response_feature_keys=tuple(
            payload.get(
                "response_feature_keys",
                payload.get(
                    "rollout_scalar_keys",
                    payload.get("trajectory_scalar_keys", ()),
                ),
            )
        ),
        derived_response_feature_keys=tuple(
            payload.get(
                "derived_response_feature_keys",
                payload.get("trajectory_derived_scalar_keys", ()),
            )
        ),
        model=EstimatorModelConfig(
            alpha=_estimator_alpha(estimator),
            clip_min=clip_min,
            clip_max=clip_max,
            feature_dim=feature_dim,
        ),
    )


class SingleTrajectoryFeatureBuilder:
    """Build the exact three inputs consumed by a trajectory estimator."""

    def __init__(self, config: FeatureBuilderConfig) -> None:
        self.config = config

    @staticmethod
    def _hidden_vector(values: np.ndarray | Sequence[float]) -> np.ndarray:
        hidden = np.asarray(values, dtype=np.float32)
        if hidden.ndim == 1:
            return hidden
        if hidden.ndim == 2:
            return hidden.mean(axis=0, dtype=np.float32)
        raise ValueError(f"Expected pooled or token hidden states, got {hidden.shape}")

    def build_inputs(
        self,
        *,
        prompt_hidden: np.ndarray | Sequence[float],
        response_hidden: np.ndarray | Sequence[float],
        generated_text: str = "",
        response_ids: Sequence[int] = (),
        tokenizer: Any | None = None,
        rollout_features: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        reasoning_text, answer_text = _reasoning_and_answer(generated_text)
        output_length = len(response_ids)
        output_unique = _unique_token_ratio(generated_text)
        reasoning_unique = _unique_token_ratio(reasoning_text)
        feature_map = {
            key: float(value)
            for key, value in rollout_features.items()
            if _coerce_float(value) is not None
        }
        builtin_values = {
            "output_length": float(output_length),
            "think_tokens": float(_count_text_tokens(tokenizer, reasoning_text)),
            "answer_tokens": float(_count_text_tokens(tokenizer, answer_text)),
            "has_complete_answer": 1.0 if answer_text else 0.0,
            "has_reasoning_content": 1.0 if reasoning_text else 0.0,
            "output_unique_token_ratio": output_unique,
            "answer_unique_token_ratio": _unique_token_ratio(answer_text),
            "output_repetition_ratio": _repetition_ratio(generated_text),
            "reasoning_repetition_ratio": _repetition_ratio(reasoning_text),
            "duplicate_line_ratio": _duplicate_line_ratio(generated_text),
            "reasoning_unique_token_ratio": reasoning_unique,
        }
        feature_map.update(
            {
                key: float(value)
                for key, value in builtin_values.items()
                if value is not None
            }
        )
        safe_output_length = max(float(output_length), 1.0)
        feature_map.update(
            {
                "think_ratio": feature_map.get("think_tokens", 0.0)
                / safe_output_length,
                "answer_ratio": feature_map.get("answer_tokens", 0.0)
                / safe_output_length,
                "entropy_gap_reasoning_answer": feature_map.get(
                    "reasoning_mean_token_entropy", 0.0
                )
                - feature_map.get("answer_mean_token_entropy", 0.0),
                "unique_gap_reasoning_output": feature_map.get(
                    "reasoning_unique_token_ratio", 0.0
                )
                - feature_map.get("output_unique_token_ratio", 0.0),
                "repetition_gap_reasoning_output": feature_map.get(
                    "reasoning_repetition_ratio", 0.0
                )
                - feature_map.get("output_repetition_ratio", 0.0),
                "reasoning_x_log_output_length": feature_map.get(
                    "has_reasoning_content", 0.0
                )
                * float(np.log1p(output_length)),
                "answer_entropy_gap_vs_output": feature_map.get(
                    "answer_mean_token_entropy", 0.0
                )
                - feature_map.get("output_mean_token_entropy", 0.0),
            }
        )
        record = {
            "generated_text": generated_text,
            "reasoning_content": reasoning_text,
            "answer_content": answer_text,
            "output_length": output_length,
            "token_stats": {
                "think_tokens": feature_map.get("think_tokens", 0.0),
                "answer_tokens": feature_map.get("answer_tokens", 0.0),
            },
            "rollout_features": rollout_features,
        }
        for field_path in self.config.rollout_scalars.extra_scalar_field_paths:
            numeric = _coerce_float(_nested_value(record, field_path))
            if numeric is not None:
                feature_map[_sanitize_name(field_path)] = numeric

        ordered_keys = list(self.config.rollout_scalars.scalar_keys)
        ordered_keys.extend(self.config.rollout_scalars.derived_scalar_keys)
        ordered_keys.extend(
            _sanitize_name(path)
            for path in self.config.rollout_scalars.extra_scalar_field_paths
        )
        response_features = {
            key: float(feature_map.get(key, 0.0)) for key in ordered_keys
        }
        return (
            self._hidden_vector(prompt_hidden),
            self._hidden_vector(response_hidden),
            response_features,
        )


class SingleTrajectoryEstimator:
    def __init__(
        self,
        *,
        config: EstimatorConfig,
        estimator: Any,
        prompt_hidden_projection: Any | None,
        response_hidden_projection: Any | None,
    ) -> None:
        self.config = config
        self.estimator = estimator
        self.prompt_hidden_projection = prompt_hidden_projection
        self.response_hidden_projection = response_hidden_projection

    @staticmethod
    def _project(
        values: np.ndarray | Sequence[float], projection: Any | None
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(1, -1)
        if projection is not None:
            vector = projection.transform(vector).astype(np.float32, copy=False)
        return vector.reshape(-1)

    def build_feature_vector(
        self,
        *,
        prompt_hidden: np.ndarray | Sequence[float],
        response_hidden: np.ndarray | Sequence[float],
        response_features: dict[str, float],
    ) -> np.ndarray:
        scalar_keys = list(self.config.response_feature_keys)
        scalar_keys.extend(self.config.derived_response_feature_keys)
        missing_entropy_keys = sorted(
            REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS.intersection(scalar_keys)
            - response_features.keys()
        )
        if missing_entropy_keys:
            raise ValueError(
                f"Missing actual token entropy features: {missing_entropy_keys}"
            )
        scalars = np.asarray(
            [float(response_features.get(key, 0.0)) for key in scalar_keys],
            dtype=np.float32,
        )
        vector = np.concatenate(
            [
                self._project(prompt_hidden, self.prompt_hidden_projection),
                self._project(response_hidden, self.response_hidden_projection),
                scalars,
            ]
        ).astype(np.float32, copy=False)
        expected = int(self.config.model.feature_dim)
        if expected > 0 and vector.size != expected:
            raise ValueError(
                f"Estimator feature dimension mismatch: expected {expected}, got {vector.size}"
            )
        return vector

    def predict_value(
        self,
        *,
        prompt_hidden: np.ndarray | Sequence[float],
        response_hidden: np.ndarray | Sequence[float],
        response_features: dict[str, float],
    ) -> float:
        vector = self.build_feature_vector(
            prompt_hidden=prompt_hidden,
            response_hidden=response_hidden,
            response_features=response_features,
        )
        prediction = float(
            np.asarray(self.estimator.predict(vector.reshape(1, -1))).reshape(-1)[0]
        )
        return float(
            np.clip(
                prediction,
                self.config.model.clip_min,
                self.config.model.clip_max,
            )
        )

    def predict_value_with_members(
        self,
        *,
        prompt_hidden: np.ndarray | Sequence[float],
        response_hidden: np.ndarray | Sequence[float],
        response_features: dict[str, float],
    ) -> tuple[float, list[float] | None]:
        """Return the combined prediction and optional child predictions.

        A regular estimator has no children and returns ``None`` for the second
        value. A ``MultiEstimatorRegressor`` returns its raw per-child outputs
        before the mean or ridge-stacker combine operation.
        """
        vector = self.build_feature_vector(
            prompt_hidden=prompt_hidden,
            response_hidden=response_hidden,
            response_features=response_features,
        ).reshape(1, -1)
        prediction = float(
            np.asarray(self.estimator.predict(vector)).reshape(-1)[0]
        )
        combined_prediction = float(
            np.clip(
                prediction,
                self.config.model.clip_min,
                self.config.model.clip_max,
            )
        )
        member_predictions = None
        if hasattr(self.estimator, "predict_children"):
            member_row = np.asarray(
                self.estimator.predict_children(vector),
                dtype=np.float32,
            ).reshape(-1)
            member_predictions = [float(value) for value in member_row]
        return combined_prediction, member_predictions

    def to_bundle(self) -> dict[str, Any]:
        if isinstance(self.estimator, MultiEstimatorRegressor):
            model_type = "multi_estimator"
        elif isinstance(self.estimator, SigmoidLinearRegressor):
            model_type = "sigmoid_bce"
        else:
            named_steps = getattr(self.estimator, "named_steps", {})
            final_model = named_steps.get("model")
            model_type = (
                "mlp"
                if type(final_model).__name__ == "MLPRegressor"
                else "ridge"
            )
        return {
            "bundle_type": "single_trajectory_estimator",
            "bundle_version": 1,
            "model_type": model_type,
            "config": self.config.to_dict(),
            "estimator": self.estimator,
            "prompt_hidden_pca": self.prompt_hidden_projection,
            "response_hidden_pca": self.response_hidden_projection,
        }


def load_estimator(path: str | Path) -> SingleTrajectoryEstimator:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("POISE requires joblib") from exc

    _register_legacy_pickle_aliases()
    bundle = joblib.load(Path(path).expanduser().resolve())
    estimator = bundle["estimator"]
    prompt_projection = bundle.get("prompt_hidden_pca")
    response_projection = bundle.get(
        "response_hidden_pca",
        bundle.get(
            "think_end_hidden_pca",
            bundle.get("trajectory_hidden_pca", bundle.get("rollout_hidden_pca")),
        ),
    )
    payload = bundle.get("config")
    if not isinstance(payload, dict):
        raise ValueError(f"Estimator bundle {path} has no config dictionary")
    try:
        config = EstimatorConfig.from_dict(payload)
    except Exception:
        config = _legacy_config(
            payload,
            estimator=estimator,
            prompt_projection=prompt_projection,
            response_projection=response_projection,
        )
    return SingleTrajectoryEstimator(
        config=config,
        estimator=estimator,
        prompt_hidden_projection=prompt_projection,
        response_hidden_projection=response_projection,
    )


def fit_estimator(
    *,
    rows: Sequence[dict[str, Any]],
    feature_config: FeatureBuilderConfig,
    fit_config: EstimatorFitConfig,
) -> tuple[SingleTrajectoryEstimator, dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("POISE online estimator fitting requires scikit-learn") from exc

    if not rows:
        raise ValueError("Cannot fit an estimator without rows")
    prompt_matrix = np.stack(
        [np.asarray(row["prompt_hidden"], dtype=np.float32) for row in rows]
    )
    response_matrix = np.stack(
        [np.asarray(row["response_hidden"], dtype=np.float32) for row in rows]
    )

    def fit_pca(matrix: np.ndarray, requested_dim: int):
        if requested_dim <= 0:
            return None
        effective_dim = min(requested_dim, matrix.shape[0], matrix.shape[1])
        pca = PCA(
            n_components=effective_dim,
            svd_solver="randomized",
            random_state=fit_config.random_seed,
        )
        pca.fit(matrix)
        return pca

    prompt_pca = fit_pca(prompt_matrix, fit_config.prompt_hidden_pca_dim)
    response_pca = fit_pca(response_matrix, fit_config.response_hidden_pca_dim)
    response_feature_keys = tuple(feature_config.rollout_scalars.scalar_keys)
    derived_feature_keys = tuple(feature_config.rollout_scalars.derived_scalar_keys)
    all_scalar_keys = response_feature_keys + derived_feature_keys + tuple(
        _sanitize_name(path)
        for path in feature_config.rollout_scalars.extra_scalar_field_paths
    )

    feature_rows = []
    for row in rows:
        prompt = np.asarray(row["prompt_hidden"], dtype=np.float32).reshape(1, -1)
        response = np.asarray(row["response_hidden"], dtype=np.float32).reshape(1, -1)
        if prompt_pca is not None:
            prompt = prompt_pca.transform(prompt)
        if response_pca is not None:
            response = response_pca.transform(response)
        scalars = np.asarray(
            [
                float(row["response_features"].get(key, 0.0))
                for key in all_scalar_keys
            ],
            dtype=np.float32,
        )
        feature_rows.append(
            np.concatenate([prompt.reshape(-1), response.reshape(-1), scalars])
        )
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float32)
    raw_child_configs = fit_config.child_fit_configs or ()
    if isinstance(raw_child_configs, (str, bytes)):
        raise ValueError("child_fit_configs must be a list of dictionaries")
    configs_to_fit = []
    for child_index, child_payload in enumerate(raw_child_configs):
        if not isinstance(child_payload, dict):
            raise ValueError(
                f"child_fit_configs[{child_index}] must be a dictionary"
            )
        unknown = set(child_payload) - set(EstimatorFitConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Unsupported child_fit_configs[{child_index}] keys: {sorted(unknown)}"
            )
        child_payload = dict(child_payload)
        if "model_type" in child_payload:
            child_payload["model_type"] = str(
                child_payload["model_type"]
            ).strip().lower()
        child_config = replace(
            fit_config,
            **child_payload,
            child_fit_configs=(),
            multi_estimator_combine="mean",
        )
        if child_config.model_type not in {"ridge", "sigmoid_bce", "mlp"}:
            raise ValueError(
                f"Unsupported child estimator type: {child_config.model_type}"
            )
        configs_to_fit.append(child_config)
    if not configs_to_fit:
        configs_to_fit = [fit_config]

    def fit_one(config: EstimatorFitConfig):
        if config.model_type == "ridge":
            fitted = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        Ridge(
                            alpha=config.alpha,
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            )
            fitted.fit(features, targets)
            return fitted
        if config.model_type == "mlp":
            fitted = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(max(1, int(config.mlp_hidden_dim)),),
                            activation="relu",
                            solver="adam",
                            alpha=float(config.alpha),
                            batch_size=max(
                                1,
                                min(
                                    int(config.mlp_batch_size),
                                    int(features.shape[0]),
                                ),
                            ),
                            learning_rate_init=float(config.mlp_lr),
                            max_iter=max(1, int(config.mlp_epochs)),
                            random_state=int(config.random_seed),
                        ),
                    ),
                ]
            )
            fitted.fit(features, targets)
            return fitted

        import torch
        import torch.nn.functional as functional

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features).astype(np.float32, copy=False)
        clipped_targets = np.clip(
            targets,
            float(config.sigmoid_bce_eps),
            1.0 - float(config.sigmoid_bce_eps),
        ).astype(np.float32, copy=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_tensor = torch.from_numpy(scaled).to(device)
        target_tensor = torch.from_numpy(clipped_targets).to(device)
        weight = torch.zeros(
            feature_tensor.shape[1],
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        mean_target = float(
            np.clip(
                clipped_targets.mean(),
                config.sigmoid_bce_eps,
                1.0 - config.sigmoid_bce_eps,
            )
        )
        bias = torch.tensor(
            np.log(mean_target / (1.0 - mean_target)),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        optimizer = torch.optim.LBFGS(
            [weight, bias],
            lr=1.0,
            max_iter=max(1, int(config.sigmoid_bce_max_iter)),
            line_search_fn="strong_wolfe",
        )
        row_count = float(max(1, feature_tensor.shape[0]))

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = feature_tensor @ weight + bias
            loss = functional.binary_cross_entropy_with_logits(
                logits, target_tensor
            )
            if config.alpha > 0:
                loss = loss + (
                    float(config.alpha) / (2.0 * row_count)
                ) * torch.sum(weight * weight)
            loss.backward()
            return loss

        optimizer.step(closure)
        fitted = SigmoidLinearRegressor(
            mean=scaler.mean_,
            scale=np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_),
            coef=weight.detach().cpu().numpy(),
            intercept=float(bias.detach().cpu().item()),
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )
        fitted.alpha = float(config.alpha)
        return fitted

    fitted_estimators = [fit_one(config) for config in configs_to_fit]
    if len(fitted_estimators) == 1 and not raw_child_configs:
        model = fitted_estimators[0]
    else:
        stacker = None
        if fit_config.multi_estimator_combine == "ridge_stacker":
            child_predictions = np.stack(
                [
                    np.asarray(child.predict(features), dtype=np.float32).reshape(
                        -1
                    )
                    for child in fitted_estimators
                ],
                axis=1,
            )
            stacker = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        Ridge(
                            alpha=fit_config.multi_estimator_stacker_alpha,
                            random_state=fit_config.random_seed,
                        ),
                    ),
                ]
            )
            stacker.fit(child_predictions, targets)
        model = MultiEstimatorRegressor(
            estimators=fitted_estimators,
            combine_method=fit_config.multi_estimator_combine,
            stacker=stacker,
            clip_min=fit_config.clip_min,
            clip_max=fit_config.clip_max,
            alpha=fit_config.multi_estimator_stacker_alpha,
        )
    config = EstimatorConfig(
        prompt_hidden_projection=_projection_config(prompt_pca),
        response_hidden_projection=_projection_config(response_pca),
        response_feature_keys=response_feature_keys,
        derived_response_feature_keys=derived_feature_keys
        + tuple(
            _sanitize_name(path)
            for path in feature_config.rollout_scalars.extra_scalar_field_paths
        ),
        model=EstimatorModelConfig(
            alpha=fit_config.alpha,
            clip_min=fit_config.clip_min,
            clip_max=fit_config.clip_max,
            feature_dim=int(features.shape[1]),
        ),
    )
    estimator = SingleTrajectoryEstimator(
        config=config,
        estimator=model,
        prompt_hidden_projection=prompt_pca,
        response_hidden_projection=response_pca,
    )
    predictions = np.clip(
        model.predict(features), fit_config.clip_min, fit_config.clip_max
    )
    diagnostics = regression_diagnostic_values(
        predictions=predictions,
        targets=targets,
    )
    metrics = {
        "rows": diagnostics["rows"],
        "prediction_mean": diagnostics["prediction_mean"],
        "prediction_std": diagnostics["prediction_std"],
        "prediction_p10": diagnostics["prediction_p10"],
        "prediction_p50": diagnostics["prediction_p50"],
        "prediction_p90": diagnostics["prediction_p90"],
        "target_mean": diagnostics["target_mean"],
        "target_std": diagnostics["target_std"],
        "target_p10": diagnostics["target_p10"],
        "target_p50": diagnostics["target_p50"],
        "target_p90": diagnostics["target_p90"],
        "train_mae": diagnostics["target_mae"],
        "train_rmse": diagnostics["target_rmse"],
        "train_constant_brier": diagnostics["constant_brier"],
        "train_brier_skill": diagnostics["brier_skill"],
        "train_brier_skill_defined": diagnostics[
            "brier_skill_defined"
        ],
        "train_bias": diagnostics["target_bias"],
        "train_pearson": diagnostics["target_pearson"],
        "pred_clip_frac_min": float(
            np.isclose(predictions, fit_config.clip_min, atol=1e-7).mean()
        ),
        "pred_clip_frac_max": float(
            np.isclose(predictions, fit_config.clip_max, atol=1e-7).mean()
        ),
    }
    if hasattr(model, "predict_children"):
        child_predictions = np.asarray(
            model.predict_children(features),
            dtype=np.float32,
        )
        if child_predictions.ndim != 2:
            raise ValueError(
                "Multi-estimator child predictions must have shape "
                f"[rows, children], got {child_predictions.shape}"
            )
        metrics["member_count"] = float(child_predictions.shape[1])
        for member_index in range(child_predictions.shape[1]):
            member_diagnostics = regression_diagnostic_values(
                predictions=child_predictions[:, member_index],
                targets=targets,
            )
            member_prefix = f"member_{member_index}"
            metrics.update(
                {
                    f"{member_prefix}/prediction_mean": member_diagnostics[
                        "prediction_mean"
                    ],
                    f"{member_prefix}/prediction_std": member_diagnostics[
                        "prediction_std"
                    ],
                    f"{member_prefix}/train_mae": member_diagnostics[
                        "target_mae"
                    ],
                    f"{member_prefix}/train_rmse": member_diagnostics[
                        "target_rmse"
                    ],
                    f"{member_prefix}/train_constant_brier": (
                        member_diagnostics["constant_brier"]
                    ),
                    f"{member_prefix}/train_brier_skill": (
                        member_diagnostics["brier_skill"]
                    ),
                    f"{member_prefix}/train_brier_skill_defined": (
                        member_diagnostics["brier_skill_defined"]
                    ),
                    f"{member_prefix}/train_bias": member_diagnostics[
                        "target_bias"
                    ],
                    f"{member_prefix}/train_pearson": member_diagnostics[
                        "target_pearson"
                    ],
                }
            )
    else:
        metrics["member_count"] = 0.0
    return estimator, metrics
