"""POISE trainer using the bundled Ray/FSDP runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_epoch_metrics,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer

from .core import (
    SUPPORTED_TARGET_MODES,
    compute_group_targets_and_cross_baselines,
    cross_rollout_diagnostic_metrics,
    cross_rollout_pairwise_sign_matches,
    regression_diagnostic_values,
)
from .domain_estimator_bank import (
    DOMAIN_NAMES,
    STATE_FILENAME,
    DomainEstimatorBank,
    RecentDomainBuffer,
    domain_from_data_source,
)


class RayPOISETrainer(RayPPOTrainer):
    """PPO actor updates using domain-routed POISE leave-one-out baselines."""

    _THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.I | re.S)
    _ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
    _PROMPT_REWARD_LOG_ROOT_DIRNAME = "prompt_reward_logs"
    _PROMPT_REWARD_LATEST_FILENAME = "latest_prompt_reward_log.txt"
    _ESTIMATOR_UPDATE_ROOT_DIRNAME = "adaptive_estimator_updates"
    _ESTIMATOR_LATEST_UPDATE_FILENAME = "latest_adaptive_estimator_update.txt"
    _ESTIMATOR_UPDATE_META_FILENAME = "poise_adaptive_estimator_update.meta.json"
    _ESTIMATOR_UPDATE_TRAINING_DATA_FILENAME = (
        "poise_adaptive_estimator_training_data.pt"
    )
    _PHASE_STATE_FILENAME = "poise_phase_state.pt"

    def _default_local_checkpoint_root(self) -> str:
        root = str(self.config.trainer.default_local_dir)
        if not os.path.isabs(root):
            root = os.path.join(os.getcwd(), root)
        return root

    def _resolve_prompt_reward_log_dir(self) -> str:
        configured = self.config.trainer.poise.get("prompt_reward_log_dir", None)
        if configured is not None:
            path = str(configured).strip()
            if path and path.lower() not in {"none", "null", "auto"}:
                path = os.path.expanduser(path)
                if not os.path.isabs(path):
                    path = os.path.join(os.getcwd(), path)
                return path
        return os.path.join(
            self._default_local_checkpoint_root(),
            self._PROMPT_REWARD_LOG_ROOT_DIRNAME,
        )

    def _resolve_estimator_update_output_dir(self) -> str:
        configured = self.config.trainer.poise.estimator.get(
            "online_output_dir", None
        )
        if configured is not None:
            path = str(configured).strip()
            if path and path.lower() not in {"none", "null", "auto"}:
                path = os.path.expanduser(path)
                if not os.path.isabs(path):
                    path = os.path.join(os.getcwd(), path)
                return path
        return os.path.join(
            self._default_local_checkpoint_root(),
            self._ESTIMATOR_UPDATE_ROOT_DIRNAME,
        )

    def _should_save_estimator_update(self) -> bool:
        save_freq = int(self.config.trainer.poise.estimator.save_freq)
        return (
            save_freq > 0
            and hasattr(self, "_poise_bank")
            and self._poise_phase == "poise"
            and self.global_steps % save_freq == 0
        )

    def _save_estimator_update_snapshot(self) -> str:
        if not hasattr(self, "_poise_bank"):
            raise RuntimeError("Cannot save a POISE estimator before it is initialized")
        output_root = self._resolve_estimator_update_output_dir()
        snapshot_dir = os.path.join(
            output_root,
            f"global_step_{self.global_steps}",
        )
        os.makedirs(snapshot_dir, exist_ok=True)
        self._poise_bank.save(snapshot_dir)

        training_data_payload = self._poise_bank.adaptive_training_data_payload(
            global_step=self.global_steps
        )
        training_data_path = os.path.join(
            snapshot_dir,
            self._ESTIMATOR_UPDATE_TRAINING_DATA_FILENAME,
        )
        temporary_training_data_path = (
            f"{training_data_path}.tmp-{os.getpid()}"
        )
        try:
            torch.save(training_data_payload, temporary_training_data_path)
            os.replace(temporary_training_data_path, training_data_path)
        finally:
            Path(temporary_training_data_path).unlink(missing_ok=True)

        estimator_model_paths = {
            domain: os.path.join(snapshot_dir, filename)
            for domain, filename in training_data_payload["estimator_files"].items()
        }
        meta_payload = {
            key: training_data_payload[key]
            for key in (
                "schema_version",
                "global_step",
                "retrain_steps",
                "retrain_count",
                "retrain_steps_by_domain",
                "retrain_count_by_domain",
                "refit_domains_this_step",
                "buffer_rows",
                "buffer_steps",
                "buffer_step_row_counts",
                "buffer_max_rows_by_domain",
                "domain_summaries",
                "estimator_trained_from_saved_data",
            )
        }
        meta_payload.update(
            {
                key: training_data_payload[key]
                for key in ("recent_buffer_rows",)
                if key in training_data_payload
            }
        )
        meta_payload.update(
            {
                "estimator_model_paths": estimator_model_paths,
                "snapshot_saved": all(
                    os.path.isfile(path)
                    for path in estimator_model_paths.values()
                ),
                "training_data_path": training_data_path,
                "training_data_saved": os.path.isfile(training_data_path),
            }
        )
        meta_path = os.path.join(
            snapshot_dir,
            self._ESTIMATOR_UPDATE_META_FILENAME,
        )
        temporary_meta_path = f"{meta_path}.tmp-{os.getpid()}"
        try:
            with open(temporary_meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta_payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_meta_path, meta_path)
        finally:
            Path(temporary_meta_path).unlink(missing_ok=True)

        os.makedirs(output_root, exist_ok=True)
        latest_path = os.path.join(
            output_root,
            self._ESTIMATOR_LATEST_UPDATE_FILENAME,
        )
        temporary_latest_path = f"{latest_path}.tmp-{os.getpid()}"
        try:
            with open(temporary_latest_path, "w", encoding="utf-8") as handle:
                handle.write(str(self.global_steps))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_latest_path, latest_path)
        finally:
            Path(temporary_latest_path).unlink(missing_ok=True)
        print(f"[POISE] Saved adaptive estimator update to {snapshot_dir}")
        return snapshot_dir

    def _scratch_bootstrap_enabled(self) -> bool:
        return bool(self.config.trainer.poise.bootstrap.from_scratch)

    def _bootstrap_uses_rloo_batch_order(self) -> bool:
        """Whether the bootstrap keeps verl's RLOO actor-batch order.

        The POISE loop balances tokens across DP ranks right after generation,
        exactly where ``RayPPOTrainer.fit`` does, and then rebuilds the batch by
        uid and balances it a second time before the update. The second pass
        reorders an already balanced batch, so a bootstrap step that is supposed
        to reproduce an RLOO baseline ends up with a different rank assignment
        and, under ``use_dynamic_bsz`` + ``token-mean``, a different gradient.

        With this enabled the bootstrap phase keeps the first balancing pass and
        skips the regroup-and-rebalance, which is what verl does. It changes
        nothing once the run leaves the bootstrap phase.
        """
        return bool(
            getattr(self.config.trainer.poise.bootstrap, "rloo_batch_order", False)
        )

    def _estimator_runtime_kwargs(self) -> dict:
        estimator_cfg = self.config.trainer.poise.estimator
        return {
            "feature_config_path": str(
                estimator_cfg.feature_builder_config_path
            ),
            "fit_config_path": str(estimator_cfg.fit_config_path),
            "warmup_steps": int(estimator_cfg.warmup_steps),
            "buffer_max_rows": {
                domain: int(estimator_cfg.buffer_max_rows[domain])
                for domain in DOMAIN_NAMES
            },
        }

    def _set_estimator_runtime(self, runtime: DomainEstimatorBank) -> None:
        self._poise_runtime = runtime
        target_mode = runtime.fit_config.target_mode
        if target_mode not in SUPPORTED_TARGET_MODES:
            raise ValueError(
                f"Unsupported POISE target_mode={target_mode!r}; "
                f"expected one of {sorted(SUPPORTED_TARGET_MODES)}"
            )
        capture_spec = runtime.capture_spec
        capture_spec["think_end_token_ids"] = self.tokenizer.encode(
            "</think>", add_special_tokens=False
        )
        self._poise_capture_spec = capture_spec

    def _initialize_scratch_bootstrap(self) -> None:
        runtime = DomainEstimatorBank.runtime(**self._estimator_runtime_kwargs())
        if runtime.fit_config.target_mode != "other_rollout_correctness":
            raise ValueError(
                "Scratch POISE bootstrap requires "
                "target_mode='other_rollout_correctness'"
            )
        self._set_estimator_runtime(runtime)
        self._bootstrap_buffer = RecentDomainBuffer(runtime.buffer_max_rows)
        self._poise_phase = "bootstrap"

    def _validate_poise_config(self) -> None:
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        validation_rollouts = int(
            self.config.trainer.poise.validation_rollouts_per_prompt
        )
        effective_validation_rollouts = int(
            self.config.actor_rollout_ref.rollout.val_kwargs.n
        )
        bootstrap_cfg = self.config.trainer.poise.bootstrap
        estimator_cfg = self.config.trainer.poise.estimator
        group_size = int(estimator_cfg.group_size)
        if rollout_n < 2:
            raise ValueError("POISE requires actor_rollout_ref.rollout.n >= 2")
        if group_size != rollout_n:
            raise ValueError(
                "trainer.poise.estimator.group_size must equal "
                f"actor_rollout_ref.rollout.n ({group_size} != {rollout_n})"
            )
        if self.config.algorithm.adv_estimator not in {
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO.value,
        }:
            raise ValueError(
                "POISE keeps algorithm.adv_estimator=grpo for PPO configuration compatibility"
            )
        if self.config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("POISE hidden capture currently supports FSDP/FSDP2")
        if not isinstance(estimator_cfg.eval_on_val, bool):
            raise TypeError("trainer.poise.estimator.eval_on_val must be a boolean")
        if int(estimator_cfg.save_freq) <= 0:
            raise ValueError("trainer.poise.estimator.save_freq must be positive")
        if estimator_cfg.eval_on_val:
            if validation_rollouts < 2:
                raise ValueError(
                    "POISE estimator validation requires at least two rollouts "
                    "per prompt"
                )
            if effective_validation_rollouts != validation_rollouts:
                raise ValueError(
                    "POISE validation rollout override was not applied: "
                    f"{effective_validation_rollouts} != {validation_rollouts}"
                )
        if not isinstance(bootstrap_cfg.from_scratch, bool):
            raise TypeError("trainer.poise.bootstrap.from_scratch must be a boolean")
        if not isinstance(getattr(bootstrap_cfg, "rloo_batch_order", False), bool):
            raise TypeError(
                "trainer.poise.bootstrap.rloo_batch_order must be a boolean"
            )
        bootstrap_steps = int(bootstrap_cfg.steps)
        if bootstrap_cfg.from_scratch:
            if bootstrap_steps <= 0:
                raise ValueError(
                    "Scratch POISE bootstrap requires bootstrap.steps > 0"
                )
            if rollout_n != 2:
                raise ValueError("Scratch POISE bootstrap currently requires rollout.n=2")
        elif bootstrap_steps != 0:
            raise ValueError(
                "bootstrap.steps must be zero when from_scratch is false"
            )

    def _initialize_estimator_bank(self) -> DomainEstimatorBank:
        estimator_cfg = self.config.trainer.poise.estimator
        model_paths = {
            domain: str(estimator_cfg.model_paths[domain]) for domain in DOMAIN_NAMES
        }
        missing = [path for path in model_paths.values() if not Path(path).expanduser().exists()]
        if missing:
            formatted = "\n  ".join(missing)
            raise FileNotFoundError(
                "POISE initial estimator files are missing:\n  " + formatted
            )
        bank = DomainEstimatorBank(
            model_paths=model_paths,
            **self._estimator_runtime_kwargs(),
        )
        self._set_estimator_runtime(bank)
        return bank

    @staticmethod
    def _stable_prompt_uids(batch: DataProto) -> np.ndarray:
        uids = []
        for input_ids in batch.batch["input_ids"]:
            prompt_bytes = input_ids.detach().cpu().contiguous().numpy().tobytes()
            uids.append(hashlib.sha256(prompt_bytes).hexdigest())
        return np.asarray(uids, dtype=object)

    @staticmethod
    def _normalize_non_tensor_rows(batch: DataProto) -> DataProto:
        """Keep every non-tensor field one-dimensional across mini-batches.

        ``rl_dataset.collate_fn`` uses ``np.array(..., dtype=object)``.  NumPy
        still infers extra dimensions when every nested value in a batch has
        the same length, so a field such as ``solutions`` can be ``(32, 1)``
        for one parquet file and ``(32,)`` at a heterogeneous file boundary.
        POISE accumulates several generation mini-batches before rollout, and
        ``DataProto.concat`` cannot concatenate those differing ranks.

        Store one Python object per example while preserving the leading batch
        dimension.  This is intentionally local to the POISE prompt buffer;
        shared dataset collation and other trainers remain unchanged.
        """
        batch_size = len(batch.batch)
        normalized: dict[str, np.ndarray] = {}
        for key, values in batch.non_tensor_batch.items():
            array = np.asarray(values, dtype=object)
            if array.ndim == 0 or array.shape[0] != batch_size:
                raise ValueError(
                    "POISE prompt non-tensor field has no aligned batch "
                    f"dimension: key={key!r}, shape={array.shape}, "
                    f"batch_size={batch_size}"
                )
            if array.ndim == 1:
                normalized[key] = array
                continue

            row_objects = np.empty(batch_size, dtype=object)
            row_objects[:] = [
                row.tolist() if isinstance(row, np.ndarray) else row
                for row in array
            ]
            normalized[key] = row_objects

        batch.non_tensor_batch = normalized
        return batch

    @staticmethod
    def _unique_prompt_indices(
        batch: DataProto, existing_uids: set[str] | None = None
    ) -> list[int]:
        seen = set() if existing_uids is None else set(existing_uids)
        indices = []
        for index, uid in enumerate(batch.non_tensor_batch["uid"]):
            key = str(uid)
            if key in seen:
                continue
            seen.add(key)
            indices.append(index)
        return indices

    @staticmethod
    def _uid_to_indices(batch: DataProto) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, uid in enumerate(batch.non_tensor_batch["uid"]):
            groups[str(uid)].append(index)
        return groups

    @classmethod
    def _actor_batch_indices(
        cls,
        batch: DataProto,
        *,
        prompt_count: int,
        group_size: int,
        preserve_order: bool = False,
    ) -> list[int]:
        groups = cls._uid_to_indices(batch)
        malformed = {
            uid: len(indices)
            for uid, indices in groups.items()
            if len(indices) != group_size
        }
        if malformed:
            raise ValueError(f"Malformed POISE rollout groups: {list(malformed.items())[:5]}")
        if len(groups) != prompt_count:
            raise ValueError(
                f"Expected {prompt_count} prompt groups, got {len(groups)}"
            )
        if preserve_order:
            # Preserve the first token-balancing pass for RLOO compatibility.
            return list(range(len(batch.batch)))
        return [index for indices in groups.values() for index in indices]

    @staticmethod
    def _span_token_indices(
        offsets: list[tuple[int, int]], span: tuple[int, int] | None
    ) -> list[int]:
        if span is None:
            return []
        start, end = span
        return [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and token_end > start
            and token_start < end
        ]

    def _reasoning_answer_spans(
        self, generated_text: str
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        think_match = self._THINK_PATTERN.search(generated_text)
        answer_match = self._ANSWER_PATTERN.search(generated_text)
        reasoning_span = (
            (think_match.start(1), think_match.end(1)) if think_match else None
        )
        if answer_match:
            answer_span = (answer_match.start(1), answer_match.end(1))
        elif "</think>" in generated_text:
            start = generated_text.find("</think>") + len("</think>")
            while start < len(generated_text) and generated_text[start].isspace():
                start += 1
            answer_span = (
                (start, len(generated_text))
                if start < len(generated_text)
                else None
            )
        else:
            answer_span = None
        return reasoning_span, answer_span

    @staticmethod
    def _entropy_stats(
        values: np.ndarray, indices: list[int] | None = None
    ) -> dict[str, float]:
        selected = values
        if indices is not None:
            valid = [index for index in indices if 0 <= index < values.size]
            selected = values[valid] if valid else np.asarray([], dtype=np.float32)
        if selected.size == 0:
            return {"mean": 0.0, "last": 0.0, "max": 0.0, "min": 0.0}
        return {
            "mean": float(selected.mean()),
            "last": float(selected[-1]),
            "max": float(selected.max()),
            "min": float(selected.min()),
        }

    def _rollout_entropy_features(
        self,
        *,
        generated_text: str,
        response_ids: list[int],
        entropies: np.ndarray,
    ) -> dict[str, float]:
        reasoning_indices: list[int] = []
        answer_indices: list[int] = []
        use_fallback = False
        try:
            encoded = self.tokenizer(
                generated_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
                offsets = offsets[0]
            if len(token_ids) == len(response_ids):
                reasoning_span, answer_span = self._reasoning_answer_spans(
                    generated_text
                )
                normalized_offsets = [(int(start), int(end)) for start, end in offsets]
                reasoning_indices = self._span_token_indices(
                    normalized_offsets, reasoning_span
                )
                answer_indices = self._span_token_indices(
                    normalized_offsets, answer_span
                )
            else:
                use_fallback = True
        except Exception:
            use_fallback = True
        if use_fallback:
            think_match = self._THINK_PATTERN.search(generated_text)
            answer_match = self._ANSWER_PATTERN.search(generated_text)
            reasoning_text = think_match.group(1) if think_match else ""
            if answer_match:
                answer_text = answer_match.group(1)
            elif "</think>" in generated_text:
                answer_text = generated_text.split("</think>", maxsplit=1)[1].strip()
            else:
                answer_text = ""
            reasoning_count = len(
                self.tokenizer.encode(reasoning_text, add_special_tokens=False)
            )
            answer_count = len(
                self.tokenizer.encode(answer_text, add_special_tokens=False)
            )
            total_tokens = len(response_ids)
            reasoning_count = min(reasoning_count, total_tokens)
            answer_count = min(answer_count, total_tokens)
            reasoning_indices = list(range(reasoning_count))
            answer_indices = list(
                range(max(0, total_tokens - answer_count), total_tokens)
            )

        output = self._entropy_stats(entropies)
        reasoning = self._entropy_stats(entropies, reasoning_indices)
        answer = self._entropy_stats(entropies, answer_indices)
        return {
            "output_mean_token_entropy": output["mean"],
            "output_last_token_entropy": output["last"],
            "output_max_token_entropy": output["max"],
            "output_min_token_entropy": output["min"],
            "reasoning_mean_token_entropy": reasoning["mean"],
            "reasoning_last_token_entropy": reasoning["last"],
            "reasoning_max_token_entropy": reasoning["max"],
            "reasoning_min_token_entropy": reasoning["min"],
            "answer_mean_token_entropy": answer["mean"],
            "answer_last_token_entropy": answer["last"],
            "answer_max_token_entropy": answer["max"],
            "answer_min_token_entropy": answer["min"],
        }

    def _build_poise_feature_rows(
        self,
        *,
        batch: DataProto,
        entropies: torch.Tensor,
    ) -> tuple[list[str], list[dict]]:
        """Build estimator inputs without requiring a fitted estimator."""
        required = {"estimator_prompt_hidden", "estimator_response_hidden"}
        missing = required - set(batch.batch.keys())
        if missing:
            raise KeyError(f"POISE hidden capture did not return tensors: {sorted(missing)}")
        if "data_source" not in batch.non_tensor_batch:
            raise KeyError("POISE domain routing requires data_source")

        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        domains = [
            domain_from_data_source(source)
            for source in batch.non_tensor_batch["data_source"]
        ]
        row_inputs: list[dict] = []
        for index, domain in enumerate(domains):
            valid_mask = response_mask[index].bool().detach().cpu()
            response_ids = responses[index].detach().cpu()[valid_mask].tolist()
            generated_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            )
            entropy_values = (
                entropies[index].detach().cpu()[valid_mask].numpy().astype(np.float32)
            )
            rollout_features = self._rollout_entropy_features(
                generated_text=generated_text,
                response_ids=response_ids,
                entropies=entropy_values,
            )
            prompt_hidden, response_hidden, response_features = (
                self._poise_runtime.feature_builder.build_inputs(
                    prompt_hidden=batch.batch["estimator_prompt_hidden"][index]
                    .detach()
                    .cpu()
                    .numpy(),
                    response_hidden=batch.batch["estimator_response_hidden"][index]
                    .detach()
                    .cpu()
                    .numpy(),
                    generated_text=generated_text,
                    response_ids=response_ids,
                    tokenizer=self.tokenizer,
                    rollout_features=rollout_features,
                )
            )
            row_inputs.append(
                {
                    "domain": domain,
                    "prompt_hidden": prompt_hidden,
                    "response_hidden": response_hidden,
                    "response_features": response_features,
                }
            )

        return domains, row_inputs

    def _predict_poise_values(
        self,
        *,
        batch: DataProto,
        entropies: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[list[float] | None],
        list[str],
        list[dict],
    ]:
        """Predict values once for a generated batch using domain routing."""
        domains, row_inputs = self._build_poise_feature_rows(
            batch=batch,
            entropies=entropies,
        )
        predictions = torch.zeros(len(batch.batch), dtype=torch.float32)
        member_value_predictions: list[list[float] | None] = [
            None
        ] * len(batch.batch)
        for index, (domain, row) in enumerate(zip(domains, row_inputs, strict=True)):
            prediction, member_predictions = self._poise_bank.estimator_for(
                domain
            ).predict_value_with_members(
                prompt_hidden=row["prompt_hidden"],
                response_hidden=row["response_hidden"],
                response_features=row["response_features"],
            )
            predictions[index] = prediction
            member_value_predictions[index] = member_predictions

        return predictions, member_value_predictions, domains, row_inputs

    def _compute_poise_advantages(
        self,
        *,
        batch: DataProto,
        reward_sums: torch.Tensor,
        entropies: torch.Tensor,
    ) -> tuple[dict[str, list[dict]], dict[str, float]]:
        (
            predictions,
            member_value_predictions,
            domains,
            row_inputs,
        ) = self._predict_poise_values(batch=batch, entropies=entropies)
        group_size = int(self.config.trainer.poise.estimator.group_size)

        uid_to_indices = self._uid_to_indices(batch)
        for uid, indices in uid_to_indices.items():
            uid_domains = {domains[index] for index in indices}
            if len(uid_domains) != 1:
                raise ValueError(
                    f"POISE uid={uid!r} crosses domains: {sorted(uid_domains)}"
                )

        targets, baselines = compute_group_targets_and_cross_baselines(
            reward_sums=reward_sums,
            value_predictions=predictions,
            uid_to_indices=uid_to_indices,
            group_size=group_size,
            target_mode=self._poise_bank.fit_config.target_mode,
        )
        raw_advantages = reward_sums.to(torch.float32) - baselines
        batch.batch["poise_value_predictions"] = predictions
        batch.batch["poise_targets"] = targets
        batch.batch["poise_raw_advantages"] = raw_advantages
        batch.batch["poise_baselines"] = baselines
        if any(values is not None for values in member_value_predictions):
            batch.non_tensor_batch["poise_member_value_predictions"] = np.asarray(
                member_value_predictions,
                dtype=object,
            )

        rows_by_domain = {domain: [] for domain in DOMAIN_NAMES}
        metrics: dict[str, float] = {}
        for index, row in enumerate(row_inputs):
            row["uid"] = str(batch.non_tensor_batch["uid"][index])
            row["global_step"] = int(getattr(self, "global_steps", 0))
            row["target"] = float(targets[index].item())
            row["reward"] = float(reward_sums[index].item())
            row["prediction"] = float(predictions[index].item())
            row["cross_baseline"] = float(baselines[index].item())
            row["advantage"] = float(raw_advantages[index].item())
            if member_value_predictions[index] is not None:
                row["member_predictions"] = list(
                    member_value_predictions[index]
                )
            rows_by_domain[row["domain"]].append(row)

        for domain in DOMAIN_NAMES:
            indices = [
                index for index, row_domain in enumerate(domains) if row_domain == domain
            ]
            prefix = f"poise/online/{domain}"
            metrics[f"{prefix}/rows"] = float(len(indices))
            if not indices:
                continue
            prediction_values = (
                predictions[indices].detach().cpu().numpy().astype(np.float32)
            )
            target_values = (
                targets[indices].detach().cpu().numpy().astype(np.float32)
            )
            reward_values = (
                reward_sums[indices]
                .detach()
                .to(torch.float32)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            baseline_values = (
                baselines[indices].detach().cpu().numpy().astype(np.float32)
            )
            advantage_values = (
                raw_advantages[indices]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            regression_metrics = regression_diagnostic_values(
                predictions=prediction_values,
                targets=target_values,
            )
            metrics.update(
                {
                    f"{prefix}/{name}": value
                    for name, value in regression_metrics.items()
                }
            )
            metrics[f"{prefix}/advantage_mean"] = float(
                advantage_values.mean()
            )
            metrics[f"{prefix}/advantage_std"] = float(
                advantage_values.std()
            )
            metrics[f"{prefix}/reward_mean"] = float(reward_values.mean())
            metrics[f"{prefix}/reward_std"] = float(reward_values.std())

            estimator_config = self._poise_bank.estimator_for(domain).config
            metrics[f"{prefix}/pred_clip_frac_min"] = float(
                np.isclose(
                    prediction_values,
                    estimator_config.model.clip_min,
                    atol=1e-7,
                ).mean()
            )
            metrics[f"{prefix}/pred_clip_frac_max"] = float(
                np.isclose(
                    prediction_values,
                    estimator_config.model.clip_max,
                    atol=1e-7,
                ).mean()
            )

            local_uid_to_indices: dict[str, list[int]] = defaultdict(list)
            for local_index, original_index in enumerate(indices):
                local_uid_to_indices[
                    str(batch.non_tensor_batch["uid"][original_index])
                ].append(local_index)
            sign_matches = cross_rollout_pairwise_sign_matches(
                reward_sums=torch.from_numpy(reward_values),
                value_predictions=torch.from_numpy(prediction_values),
                uid_to_indices=local_uid_to_indices,
                group_size=group_size,
                target_mode=self._poise_bank.fit_config.target_mode,
            )
            metrics[f"{prefix}/pairwise_sign_acc"] = (
                float(np.mean(sign_matches)) if sign_matches else 0.0
            )
            metrics.update(
                cross_rollout_diagnostic_metrics(
                    rewards=reward_values,
                    value_predictions=prediction_values,
                    targets=target_values,
                    cross_baselines=baseline_values,
                    advantages=advantage_values,
                    group_size=group_size,
                    prefix=f"{prefix}/cross_rollout",
                )
            )

            domain_member_rows = [
                member_value_predictions[index] for index in indices
            ]
            present_member_rows = [
                values
                for values in domain_member_rows
                if values is not None
            ]
            if present_member_rows:
                if len(present_member_rows) != len(domain_member_rows):
                    raise ValueError(
                        f"POISE {domain} estimator returned members for only "
                        "part of a domain batch"
                    )
                member_matrix = np.asarray(
                    present_member_rows,
                    dtype=np.float32,
                )
                if member_matrix.ndim != 2:
                    raise ValueError(
                        f"POISE {domain} member predictions must have shape "
                        f"[rows, members], got {member_matrix.shape}"
                    )
                metrics[f"{prefix}/member_count"] = float(
                    member_matrix.shape[1]
                )
                for member_index in range(member_matrix.shape[1]):
                    member_metrics = regression_diagnostic_values(
                        predictions=member_matrix[:, member_index],
                        targets=target_values,
                    )
                    metrics.update(
                        {
                            f"{prefix}/member_{member_index}/{name}": value
                            for name, value in member_metrics.items()
                        }
                    )
            else:
                metrics[f"{prefix}/member_count"] = 0.0

        metrics["poise/advantage/mean"] = float(raw_advantages.mean().item())
        metrics["poise/advantage/std"] = float(
            raw_advantages.std(unbiased=False).item()
        )
        metrics["poise/baseline/mean"] = float(baselines.mean().item())
        metrics["poise/reward/out_of_unit_range_frac"] = float(
            ((reward_sums < 0) | (reward_sums > 1)).to(torch.float32).mean().item()
        )
        return rows_by_domain, metrics

    def _compute_bootstrap_rloo_advantages(
        self,
        *,
        batch: DataProto,
        reward_sums: torch.Tensor,
        entropies: torch.Tensor,
    ) -> tuple[dict[str, list[dict]], dict[str, float]]:
        domains, row_inputs = self._build_poise_feature_rows(
            batch=batch,
            entropies=entropies,
        )
        uid_to_indices = self._uid_to_indices(batch)
        group_size = int(self.config.actor_rollout_ref.rollout.n)
        for uid, indices in uid_to_indices.items():
            if len(indices) != group_size:
                raise ValueError(
                    f"Bootstrap uid={uid!r} has {len(indices)} rows, "
                    f"expected {group_size}"
                )
            uid_domains = {domains[index] for index in indices}
            if len(uid_domains) != 1:
                raise ValueError(
                    f"Bootstrap uid={uid!r} crosses domains: "
                    f"{sorted(uid_domains)}"
                )

        targets, _ = compute_group_targets_and_cross_baselines(
            reward_sums=reward_sums,
            value_predictions=torch.zeros_like(reward_sums),
            uid_to_indices=uid_to_indices,
            group_size=group_size,
            target_mode=self._poise_runtime.fit_config.target_mode,
        )
        raw_advantages = reward_sums.to(torch.float32) - targets
        batch.batch["poise_raw_advantages"] = raw_advantages

        rows_by_domain = {domain: [] for domain in DOMAIN_NAMES}
        for index, row in enumerate(row_inputs):
            row["uid"] = str(batch.non_tensor_batch["uid"][index])
            row["global_step"] = int(getattr(self, "global_steps", 0))
            row["target"] = float(targets[index].item())
            row["reward"] = float(reward_sums[index].item())
            rows_by_domain[row["domain"]].append(row)

        metrics = {
            "poise/bootstrap/advantage_mean": float(raw_advantages.mean().item()),
            "poise/bootstrap/advantage_std": float(
                raw_advantages.std(unbiased=False).item()
            ),
        }
        for domain in DOMAIN_NAMES:
            metrics[f"poise/bootstrap/{domain}/rows_generated"] = float(
                len(rows_by_domain[domain])
            )
        return rows_by_domain, metrics

    def _bootstrap_buffer_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for domain in DOMAIN_NAMES:
            prefix = f"poise/bootstrap/buffer/{domain}"
            metrics[f"{prefix}/rows"] = float(
                self._bootstrap_buffer.row_counts[domain]
            )
            metrics[f"{prefix}/max_rows"] = float(
                self._bootstrap_buffer.buffer_max_rows[domain]
            )
        return metrics

    def _fit_bootstrap_estimators(self) -> dict[str, float]:
        rows_by_domain = self._bootstrap_buffer.rows_by_domain()
        bank, metrics = DomainEstimatorBank.from_rows(
            rows_by_domain=rows_by_domain,
            observed_steps=self.global_steps,
            **self._estimator_runtime_kwargs(),
        )
        self._poise_bank = bank
        self._set_estimator_runtime(bank)
        self._poise_phase = "poise"
        metrics["poise/bootstrap/completed_step"] = float(self.global_steps)
        print(
            "[POISE] Scratch bootstrap complete at step "
            f"{self.global_steps}: "
            + ", ".join(
                f"{domain}={len(rows_by_domain[domain])} rows"
                for domain in DOMAIN_NAMES
            )
        )
        return metrics

    def _estimator_validation_batch_rows(
        self,
        *,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        prompt_offset: int,
    ) -> list[dict]:
        """Score the exact rollouts already generated by regular validation."""
        rollout_count = int(
            self.config.trainer.poise.validation_rollouts_per_prompt
        )
        batch_size = len(batch.batch)
        if batch_size % rollout_count != 0:
            raise ValueError(
                "POISE validation batch must contain complete prompt groups: "
                f"rows={batch_size}, rollouts_per_prompt={rollout_count}"
            )
        if reward_tensor.shape[0] != batch_size:
            raise ValueError(
                "POISE validation rewards and rollouts have different row counts: "
                f"{reward_tensor.shape[0]} != {batch_size}"
            )

        response_mask = compute_response_mask(batch)
        had_response_mask = "response_mask" in batch.batch
        previous_response_mask = (
            batch.batch["response_mask"] if had_response_mask else None
        )
        had_capture_spec = "estimator_hidden_capture" in batch.meta_info
        previous_capture_spec = batch.meta_info.get("estimator_hidden_capture")
        batch.batch["response_mask"] = response_mask
        batch.meta_info["estimator_hidden_capture"] = dict(
            self._poise_capture_spec
        )
        try:
            log_prob_output = self.actor_rollout_wg.compute_log_prob(batch)
        finally:
            if had_response_mask:
                batch.batch["response_mask"] = previous_response_mask
            else:
                batch.batch.pop("response_mask")
            if had_capture_spec:
                batch.meta_info["estimator_hidden_capture"] = (
                    previous_capture_spec
                )
            else:
                batch.meta_info.pop("estimator_hidden_capture", None)

        required_output_keys = {
            "entropys",
            "estimator_prompt_hidden",
            "estimator_response_hidden",
        }
        missing = required_output_keys - set(log_prob_output.batch.keys())
        if missing:
            raise KeyError(
                "POISE validation hidden capture did not return tensors: "
                f"{sorted(missing)}"
            )
        prediction_batch = DataProto.from_dict(
            tensors={
                "responses": batch.batch["responses"],
                "response_mask": response_mask,
                "estimator_prompt_hidden": log_prob_output.batch[
                    "estimator_prompt_hidden"
                ],
                "estimator_response_hidden": log_prob_output.batch[
                    "estimator_response_hidden"
                ],
            },
            non_tensors={
                "data_source": np.asarray(
                    batch.non_tensor_batch["data_source"], dtype=object
                )
            },
        )
        (
            predictions,
            member_predictions,
            domains,
            _,
        ) = self._predict_poise_values(
            batch=prediction_batch,
            entropies=log_prob_output.batch["entropys"],
        )
        rewards = (
            reward_tensor.sum(dim=-1)
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        prediction_values = predictions.detach().cpu().numpy()

        rows: list[dict] = []
        for group_start in range(0, batch_size, rollout_count):
            group_stop = group_start + rollout_count
            group_domains = set(domains[group_start:group_stop])
            if len(group_domains) != 1:
                raise ValueError(
                    "POISE validation prompt crosses estimator domains: "
                    f"{sorted(group_domains)}"
                )
            group_rewards = rewards[group_start:group_stop]
            reward_sum = float(group_rewards.sum())
            prompt_id = prompt_offset + group_start // rollout_count
            for index in range(group_start, group_stop):
                local_index = index - group_start
                row = {
                    "domain": domains[index],
                    "prompt_id": prompt_id,
                    "prediction": float(prediction_values[index]),
                    "reward": float(rewards[index]),
                    "loo_target": float(
                        (reward_sum - float(group_rewards[local_index]))
                        / (rollout_count - 1)
                    ),
                }
                if member_predictions[index] is not None:
                    row["member_predictions"] = list(
                        member_predictions[index]
                    )
                rows.append(row)
        return rows

    @staticmethod
    def _estimator_validation_metrics(
        rows: list[dict],
        *,
        metric_root: str = "val/poise_estimator",
    ) -> dict[str, float]:
        """Aggregate response-LOO and prompt-mean estimator diagnostics."""
        metrics: dict[str, float] = {}
        for scope in ("all", *DOMAIN_NAMES):
            scope_rows = (
                rows
                if scope == "all"
                else [row for row in rows if row["domain"] == scope]
            )
            if not scope_rows:
                continue
            prefix = f"{metric_root}/{scope}"
            predictions = np.asarray(
                [row["prediction"] for row in scope_rows], dtype=np.float32
            )
            loo_targets = np.asarray(
                [row["loo_target"] for row in scope_rows], dtype=np.float32
            )
            response_metrics = regression_diagnostic_values(
                predictions=predictions,
                targets=loo_targets,
            )
            metrics.update(
                {
                    f"{prefix}/response_loo/{name}": value
                    for name, value in response_metrics.items()
                }
            )

            prompt_rows: dict[int, list[dict]] = defaultdict(list)
            for row in scope_rows:
                prompt_rows[int(row["prompt_id"])].append(row)
            prompt_predictions = np.asarray(
                [
                    np.mean([row["prediction"] for row in group])
                    for group in prompt_rows.values()
                ],
                dtype=np.float32,
            )
            prompt_rewards = np.asarray(
                [
                    np.mean([row["reward"] for row in group])
                    for group in prompt_rows.values()
                ],
                dtype=np.float32,
            )
            prompt_metrics = regression_diagnostic_values(
                predictions=prompt_predictions,
                targets=prompt_rewards,
            )
            metrics.update(
                {
                    f"{prefix}/prompt_mean/{name}": value
                    for name, value in prompt_metrics.items()
                }
            )
            metrics[f"{prefix}/num_responses"] = float(len(scope_rows))
            metrics[f"{prefix}/num_prompts"] = float(len(prompt_rows))

            member_rows = [
                row.get("member_predictions") for row in scope_rows
            ]
            if all(values is not None for values in member_rows):
                member_matrix = np.asarray(member_rows, dtype=np.float32)
                if member_matrix.ndim != 2:
                    raise ValueError(
                        "POISE validation estimator members must have shape "
                        f"[rows, members], got {member_matrix.shape}"
                    )
                metrics[f"{prefix}/member_count"] = float(
                    member_matrix.shape[1]
                )
                for member_index in range(member_matrix.shape[1]):
                    member_metrics = regression_diagnostic_values(
                        predictions=member_matrix[:, member_index],
                        targets=loo_targets,
                    )
                    metrics.update(
                        {
                            f"{prefix}/member_{member_index}/response_loo/{name}": value
                            for name, value in member_metrics.items()
                        }
                    )
            else:
                metrics[f"{prefix}/member_count"] = 0.0
        return metrics

    @staticmethod
    def _compact_validation_metrics(
        metrics: dict[str, object],
    ) -> dict[str, object]:
        """Keep one canonical validation variable per data source.

        The shared validation stack exposes the same rule-based result as
        ``acc``, ``reward``, and ``score``.  Retaining all three expands every
        bootstrap statistic threefold in W&B without adding information.  Use
        accuracy when it is available, otherwise fall back to reward and then
        score.  Other auxiliary variables are preserved.
        """
        candidate_variables = ("acc", "reward", "score")
        key_variables: dict[str, tuple[str, str]] = {}
        variables_by_source: dict[str, set[str]] = defaultdict(set)

        for key in metrics:
            if not key.startswith(("val-core/", "val-aux/")):
                continue
            for variable in candidate_variables:
                marker = f"/{variable}/"
                if marker not in key:
                    continue
                source_prefix, _ = key.split(marker, maxsplit=1)
                source = source_prefix.split("/", maxsplit=1)[1]
                key_variables[key] = (source, variable)
                variables_by_source[source].add(variable)
                break

        canonical_by_source = {
            source: next(
                variable
                for variable in candidate_variables
                if variable in variables
            )
            for source, variables in variables_by_source.items()
        }
        return {
            key: value
            for key, value in metrics.items()
            if key not in key_variables
            or key_variables[key][1]
            == canonical_by_source[key_variables[key][0]]
        }

    @classmethod
    def _compact_wandb_metrics(
        cls,
        metrics: dict[str, object],
    ) -> dict[str, object]:
        """Keep one W&B series for each POISE quantity.

        The unfiltered dictionaries deliberately contain detailed diagnostics
        for local debugging and checkpoint analysis.  Several of those values
        are exact aliases, invariants, or configuration constants, however,
        and logging all of them makes the W&B workspace needlessly difficult
        to navigate.  This compactor only affects logger output; prompt logs,
        estimator snapshots, and the underlying computations stay intact.
        """
        compacted = cls._compact_validation_metrics(metrics)

        def values_match(left_key: str, right_key: str) -> bool:
            """Compare scalar aliases without hiding genuinely different data."""
            if left_key not in compacted or right_key not in compacted:
                return False
            try:
                return bool(
                    np.isclose(
                        float(compacted[left_key]),
                        float(compacted[right_key]),
                        equal_nan=True,
                    )
                )
            except (TypeError, ValueError):
                return compacted[left_key] == compacted[right_key]

        # This policy is intentionally scoped to the ``poise/`` namespace.
        # Shared PPO and validation metrics are governed by their own loggers.
        dropped: set[str] = set()

        for key, value in compacted.items():
            if key.startswith("poise/prompt_reward_log/"):
                # These only repeat the number of rows written to the local
                # prompt log.  The log itself remains fully preserved.
                dropped.add(key)
                continue

            if key.startswith("poise/online/"):
                suffix = key.split("/", maxsplit=3)[-1]
                leaf = key.rsplit("/", maxsplit=1)[-1]
                domain_prefix = "/".join(key.split("/")[:3])

                # Direct aliases of the corresponding target_* diagnostics.
                if "/online_target_" in key:
                    dropped.add(key)
                    continue

                # For N=2, cross_rollout is a verbose debug view of the same
                # reward/target and prediction/baseline pairs reported at the
                # domain root.  Its invariants remain in prompt logs, but do
                # not need one W&B chart apiece.
                if "/cross_rollout/" in key:
                    group_size_key = (
                        f"{domain_prefix}/cross_rollout/group_size"
                    )
                    try:
                        pairwise_rollout = (
                            float(compacted[group_size_key]) == 2.0
                        )
                    except (KeyError, TypeError, ValueError):
                        pairwise_rollout = False
                    if pairwise_rollout:
                        dropped.add(key)
                        continue

                # Quantiles add six near-identical charts per domain.  Keep
                # mean/std in W&B; the complete distributions remain available
                # in per-step prompt logs and estimator snapshots.
                if leaf in {
                    "prediction_p10",
                    "prediction_p50",
                    "prediction_p90",
                    "target_p10",
                    "target_p50",
                    "target_p90",
                }:
                    dropped.add(key)
                    continue

                # Counts are exactly fraction * rows; retain the normalized
                # fractions, which are comparable across domains and steps.
                if suffix.startswith("pred_clip_") and suffix.endswith("_count"):
                    dropped.add(key)
                    continue

                # The LOO target is a permutation/average of the same rewards,
                # so the domain-level mean is identical.  Guard with a value
                # comparison for safety if a future target mode changes this.
                if suffix in {"reward_mean", "reward_std"}:
                    target_suffix = suffix.replace("reward_", "target_")
                    if values_match(key, f"{domain_prefix}/{target_suffix}"):
                        dropped.add(key)
                        continue

                # With the N=2 leave-one-out baseline, domain advantage_mean is
                # exactly -target_bias.  Retain bias plus advantage_std.
                if suffix == "advantage_mean":
                    bias_key = f"{domain_prefix}/target_bias"
                    if bias_key in compacted:
                        try:
                            if np.isclose(
                                float(value),
                                -float(compacted[bias_key]),
                                equal_nan=True,
                            ):
                                dropped.add(key)
                                continue
                        except (TypeError, ValueError):
                            pass

                if suffix.endswith("constant_brier") or suffix.endswith(
                    "brier_skill_defined"
                ):
                    # constant_brier == target_std**2; the second key is only
                    # a defined/undefined flag for the retained brier_skill.
                    dropped.add(key)
                    continue

            if key.startswith("poise/estimator/"):
                parts = key.split("/")
                if len(parts) >= 4 and parts[2] in DOMAIN_NAMES:
                    domain = parts[2]
                    suffix = "/".join(parts[3:])
                    estimator_prefix = f"poise/estimator/{domain}"

                    if suffix == "buffer_rows" and values_match(
                        key, f"{estimator_prefix}/recent_buffer_rows"
                    ):
                        dropped.add(key)
                        continue
                    if suffix == "rows" and values_match(
                        key, f"{estimator_prefix}/fit_rows"
                    ):
                        dropped.add(key)
                        continue
                    if suffix == "rows_added" and values_match(
                        key, f"poise/online/{domain}/rows"
                    ):
                        dropped.add(key)
                        continue
                    if suffix == "buffer_max_rows":
                        # These are fixed W&B config values, not observations.
                        dropped.add(key)
                        continue
                    if suffix in {"observed_steps", "buffer_steps"}:
                        # observed_steps follows the W&B x-axis, while
                        # buffer_steps follows recent_buffer_rows.  The
                        # retrain counter and canonical row counts remain.
                        dropped.add(key)
                        continue
                    if suffix.rsplit("/", maxsplit=1)[-1] in {
                        "prediction_p10",
                        "prediction_p50",
                        "prediction_p90",
                        "target_p10",
                        "target_p50",
                        "target_p90",
                    }:
                        dropped.add(key)
                        continue
                    if suffix.endswith("train_constant_brier") or suffix.endswith(
                        "train_brier_skill_defined"
                    ):
                        dropped.add(key)
                        continue

            # A zero member count simply means that the configured estimator is
            # not an ensemble.  The estimator type already lives in W&B config.
            if key.startswith("poise/") and key.endswith("/member_count"):
                try:
                    if float(value) == 0.0:
                        dropped.add(key)
                except (TypeError, ValueError):
                    pass

        return {
            key: value for key, value in compacted.items() if key not in dropped
        }

    def _dump_generations(
        self,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
        metadata=None,
        generation_config=None,
        record_type="rollout",
    ):
        """Include raw POISE estimator diagnostics in validation JSONL rows."""
        validation_rows = getattr(
            self,
            "_poise_validation_dump_rows",
            None,
        )
        if record_type == "validation" and validation_rows is not None:
            if len(validation_rows) != len(inputs):
                raise ValueError(
                    "POISE validation estimator rows and generation dump rows "
                    "have different lengths: "
                    f"{len(validation_rows)} != {len(inputs)}"
                )
            estimator_metadata = {
                "estimator_prediction": [
                    float(row["prediction"]) for row in validation_rows
                ],
                "estimator_loo_target": [
                    float(row["loo_target"]) for row in validation_rows
                ],
                "estimator_error": [
                    float(row["prediction"] - row["loo_target"])
                    for row in validation_rows
                ],
                "estimator_domain": [
                    str(row["domain"]) for row in validation_rows
                ],
                "estimator_member_predictions": [
                    row.get("member_predictions") for row in validation_rows
                ],
            }
            metadata = dict(metadata or {})
            duplicate_keys = set(metadata).intersection(estimator_metadata)
            if duplicate_keys:
                raise ValueError(
                    "POISE validation estimator metadata would overwrite "
                    f"generation metadata: {sorted(duplicate_keys)}"
                )
            metadata.update(estimator_metadata)

        return super()._dump_generations(
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
            metadata=metadata,
            generation_config=generation_config,
            record_type=record_type,
        )

    def _validate(self):
        """Run base validation and score its generations with the POISE bank."""
        estimator_cfg = self.config.trainer.poise.estimator
        if not estimator_cfg.eval_on_val or not hasattr(self, "_poise_bank"):
            return self._compact_validation_metrics(super()._validate())

        validation_rows: list[dict] = []
        prompt_offset = 0
        original_val_reward_fn = self.val_reward_fn
        dump_rows_attribute = "_poise_validation_dump_rows"
        had_previous_dump_rows = hasattr(self, dump_rows_attribute)
        previous_dump_rows = getattr(self, dump_rows_attribute, None)

        def capturing_val_reward_fn(batch, return_dict=False):
            nonlocal prompt_offset
            result = original_val_reward_fn(batch, return_dict=return_dict)
            if not return_dict:
                raise ValueError(
                    "POISE estimator validation requires dictionary rewards"
                )
            rows = self._estimator_validation_batch_rows(
                batch=batch,
                reward_tensor=result["reward_tensor"],
                prompt_offset=prompt_offset,
            )
            validation_rows.extend(rows)
            prompt_offset += len(rows) // int(
                self.config.trainer.poise.validation_rollouts_per_prompt
            )
            return result

        self.val_reward_fn = capturing_val_reward_fn
        setattr(self, dump_rows_attribute, validation_rows)
        try:
            metrics = super()._validate()
        finally:
            self.val_reward_fn = original_val_reward_fn
            if had_previous_dump_rows:
                setattr(self, dump_rows_attribute, previous_dump_rows)
            else:
                delattr(self, dump_rows_attribute)
        metrics.update(
            self._estimator_validation_metrics(
                validation_rows,
                metric_root="val/poise_estimator",
            )
        )
        return self._compact_validation_metrics(metrics)

    def _validate_initial_estimator_for_scratch_bootstrap(self) -> dict:
        """Evaluate configured initial estimators without using them to train.

        Scratch bootstrap deliberately builds its training estimator from the
        first ``bootstrap.steps`` rollout updates.  Initial validation is the
        one place where loading the configured estimator artifacts is useful:
        it provides the step-zero point in the canonical
        ``val/poise_estimator`` series while this method removes the bank
        immediately afterward so bootstrap advantages and collected rows stay
        independent of those artifacts.
        """
        if hasattr(self, "_poise_bank"):
            raise RuntimeError(
                "Scratch initial-estimator validation requires no active POISE bank"
            )
        bootstrap_runtime = self._poise_runtime
        self._poise_bank = self._initialize_estimator_bank()
        try:
            return self._validate()
        finally:
            del self._poise_bank
            self._set_estimator_runtime(bootstrap_runtime)

    def _estimator_online_metrics(
        self,
        rows_by_domain: dict[str, list[dict]],
    ) -> dict[str, float]:
        """Aggregate pre-fit diagnostics over every rollout used by an update."""
        group_size = int(self.config.trainer.poise.estimator.group_size)
        target_mode = self._poise_bank.fit_config.target_mode
        metrics: dict[str, float] = {}
        for domain in DOMAIN_NAMES:
            rows = rows_by_domain.get(domain, [])
            prefix = f"poise/online/{domain}"
            metrics[f"{prefix}/rows"] = float(len(rows))
            if not rows:
                metrics[f"{prefix}/member_count"] = 0.0
                continue

            predictions = np.asarray(
                [row["prediction"] for row in rows],
                dtype=np.float32,
            )
            targets = np.asarray(
                [row["target"] for row in rows],
                dtype=np.float32,
            )
            rewards = np.asarray(
                [row["reward"] for row in rows],
                dtype=np.float32,
            )
            baselines = np.asarray(
                [row["cross_baseline"] for row in rows],
                dtype=np.float32,
            )
            advantages = np.asarray(
                [row["advantage"] for row in rows],
                dtype=np.float32,
            )
            regression_metrics = regression_diagnostic_values(
                predictions=predictions,
                targets=targets,
            )
            metrics.update(
                {
                    f"{prefix}/{name}": value
                    for name, value in regression_metrics.items()
                }
            )
            for name in ("mae", "rmse", "bias", "pearson"):
                metrics[f"{prefix}/online_target_{name}"] = (
                    regression_metrics[f"target_{name}"]
                )
            metrics[f"{prefix}/reward_mean"] = float(rewards.mean())
            metrics[f"{prefix}/reward_std"] = float(rewards.std())
            metrics[f"{prefix}/advantage_mean"] = float(advantages.mean())
            metrics[f"{prefix}/advantage_std"] = float(advantages.std())

            estimator_config = self._poise_bank.estimator_for(domain).config
            clip_min_mask = np.isclose(
                predictions,
                estimator_config.model.clip_min,
                atol=1e-7,
            )
            clip_max_mask = np.isclose(
                predictions,
                estimator_config.model.clip_max,
                atol=1e-7,
            )
            metrics[f"{prefix}/pred_clip_frac_min"] = float(
                clip_min_mask.mean()
            )
            metrics[f"{prefix}/pred_clip_frac_max"] = float(
                clip_max_mask.mean()
            )
            metrics[f"{prefix}/pred_clip_min_count"] = float(
                clip_min_mask.sum()
            )
            metrics[f"{prefix}/pred_clip_max_count"] = float(
                clip_max_mask.sum()
            )
            clip_min_reward_matches = clip_min_mask & np.isclose(
                rewards,
                estimator_config.model.clip_min,
                atol=1e-7,
            )
            clip_max_reward_matches = clip_max_mask & np.isclose(
                rewards,
                estimator_config.model.clip_max,
                atol=1e-7,
            )
            metrics[f"{prefix}/pred_clip_min_reward_match_count"] = float(
                clip_min_reward_matches.sum()
            )
            metrics[f"{prefix}/pred_clip_max_reward_match_count"] = float(
                clip_max_reward_matches.sum()
            )
            metrics[f"{prefix}/pred_clip_min_reward_match_frac"] = (
                float(clip_min_reward_matches.sum() / clip_min_mask.sum())
                if clip_min_mask.any()
                else 0.0
            )
            metrics[f"{prefix}/pred_clip_max_reward_match_frac"] = (
                float(clip_max_reward_matches.sum() / clip_max_mask.sum())
                if clip_max_mask.any()
                else 0.0
            )

            uid_to_indices: dict[str, list[int]] = defaultdict(list)
            for index, row in enumerate(rows):
                uid_to_indices[str(row["uid"])].append(index)
            sign_matches = cross_rollout_pairwise_sign_matches(
                reward_sums=torch.from_numpy(rewards),
                value_predictions=torch.from_numpy(predictions),
                uid_to_indices=uid_to_indices,
                group_size=group_size,
                target_mode=target_mode,
            )
            metrics[f"{prefix}/pairwise_sign_acc"] = (
                float(np.mean(sign_matches)) if sign_matches else 0.0
            )
            metrics.update(
                cross_rollout_diagnostic_metrics(
                    rewards=rewards,
                    value_predictions=predictions,
                    targets=targets,
                    cross_baselines=baselines,
                    advantages=advantages,
                    group_size=group_size,
                    prefix=f"{prefix}/cross_rollout",
                )
            )

            member_rows = [
                row.get("member_predictions") for row in rows
            ]
            present_member_rows = [
                values for values in member_rows if values is not None
            ]
            if present_member_rows:
                if len(present_member_rows) != len(member_rows):
                    raise ValueError(
                        f"Only part of the {domain} estimator rows contain "
                        "member predictions"
                    )
                member_matrix = np.asarray(
                    present_member_rows,
                    dtype=np.float32,
                )
                if member_matrix.ndim != 2:
                    raise ValueError(
                        f"POISE {domain} member predictions must have shape "
                        f"[rows, members], got {member_matrix.shape}"
                    )
                metrics[f"{prefix}/member_count"] = float(
                    member_matrix.shape[1]
                )
                for member_index in range(member_matrix.shape[1]):
                    member_metrics = regression_diagnostic_values(
                        predictions=member_matrix[:, member_index],
                        targets=targets,
                    )
                    metrics.update(
                        {
                            f"{prefix}/member_{member_index}/{name}": value
                            for name, value in member_metrics.items()
                        }
                    )
                    for name in ("mae", "rmse", "bias", "pearson"):
                        metrics[
                            f"{prefix}/member_{member_index}/"
                            f"online_target_{name}"
                        ] = member_metrics[f"target_{name}"]
            else:
                metrics[f"{prefix}/member_count"] = 0.0
        return metrics

    @staticmethod
    def _extract_prompt_text(raw_prompt_item) -> str:
        if isinstance(raw_prompt_item, str):
            return raw_prompt_item.strip()
        if isinstance(raw_prompt_item, dict):
            return str(raw_prompt_item.get("content", "")).strip()
        if isinstance(raw_prompt_item, (list, tuple)) and raw_prompt_item:
            first_turn = raw_prompt_item[0]
            if isinstance(first_turn, dict):
                return str(first_turn.get("content", "")).strip()
            return str(first_turn).strip()
        return str(raw_prompt_item).strip()

    def _decode_model_prompts(self, batch: DataProto) -> list[str]:
        if "prompts" not in batch.batch:
            raw_prompts = batch.non_tensor_batch.get("raw_prompt")
            if raw_prompts is None:
                return [""] * len(batch.batch)
            return [
                self._extract_prompt_text(raw_prompt)
                for raw_prompt in raw_prompts
            ]

        prompt_ids = batch.batch["prompts"]
        prompt_attention_mask = None
        if "attention_mask" in batch.batch:
            attention_mask = batch.batch["attention_mask"]
            if (
                attention_mask.ndim == 2
                and attention_mask.shape[0] == prompt_ids.shape[0]
            ):
                prompt_attention_mask = attention_mask[
                    :, : prompt_ids.shape[1]
                ].bool()

        decoded_prompts = []
        pad_token_id = self.tokenizer.pad_token_id
        for row_index in range(prompt_ids.shape[0]):
            row = prompt_ids[row_index].detach().cpu()
            if prompt_attention_mask is not None:
                row = row[prompt_attention_mask[row_index].detach().cpu()]
            elif pad_token_id is not None:
                row = row[row != pad_token_id]
            decoded_prompts.append(
                self.tokenizer.decode(
                    row.tolist(),
                    skip_special_tokens=False,
                ).strip()
            )
        return decoded_prompts

    @staticmethod
    def _prompt_log_float_values(
        values: torch.Tensor | np.ndarray | list | tuple,
        *,
        expected_len: int,
        name: str,
    ) -> list[float]:
        if isinstance(values, torch.Tensor):
            values_array = (
                values.detach().to(torch.float32).cpu().numpy().reshape(-1)
            )
        else:
            values_array = np.asarray(values, dtype=np.float32).reshape(-1)
        if values_array.size != expected_len:
            raise ValueError(
                f"Prompt reward log field {name!r} has {values_array.size} "
                f"rows, expected {expected_len}."
            )
        return [float(value) for value in values_array.tolist()]

    @staticmethod
    def _prompt_log_member_values(
        values: np.ndarray | list | tuple | None,
        *,
        expected_len: int,
    ) -> list[list[float] | None] | None:
        if values is None:
            return None
        if len(values) != expected_len:
            raise ValueError(
                "Prompt reward log member predictions have "
                f"{len(values)} rows, expected {expected_len}"
            )
        normalized = []
        for row in values:
            if row is None:
                normalized.append(None)
                continue
            row_values = np.asarray(row, dtype=np.float32).reshape(-1)
            normalized.append(
                [float(value) for value in row_values.tolist()]
            )
        return normalized

    def _accumulate_prompt_reward_log_rows(
        self,
        *,
        accumulator: dict[str, dict],
        order: list[str],
        batch: DataProto,
        reward_sums: torch.Tensor,
    ) -> None:
        uids = batch.non_tensor_batch.get("uid")
        data_sources = batch.non_tensor_batch.get("data_source")
        if uids is None or data_sources is None:
            raise KeyError(
                "POISE prompt reward logging requires uid and data_source"
            )

        row_count = len(batch.batch)
        model_prompts = self._decode_model_prompts(batch)
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        fields = {
            "rollout_rewards": reward_sums,
            "estimator_value_predictions": batch.batch[
                "poise_value_predictions"
            ],
            "estimator_cross_baselines": batch.batch["poise_baselines"],
            "estimator_targets": batch.batch["poise_targets"],
            "estimator_raw_advantages": batch.batch[
                "poise_raw_advantages"
            ],
        }
        field_values = {
            name: self._prompt_log_float_values(
                values,
                expected_len=row_count,
                name=name,
            )
            for name, values in fields.items()
        }
        member_values = self._prompt_log_member_values(
            batch.non_tensor_batch.get("poise_member_value_predictions"),
            expected_len=row_count,
        )

        for row_index in range(row_count):
            uid = str(uids[row_index])
            data_source = str(data_sources[row_index])
            domain = domain_from_data_source(data_source)
            if uid not in accumulator:
                order.append(uid)
                accumulator[uid] = {
                    "uid": uid,
                    "domain": domain,
                    "data_source": data_source,
                    "prompt": model_prompts[row_index],
                    "raw_prompt": (
                        self._extract_prompt_text(raw_prompts[row_index])
                        if raw_prompts is not None
                        else ""
                    ),
                    "rollout_rewards": [],
                    "estimator_value_predictions": [],
                    "estimator_cross_baselines": [],
                    "estimator_targets": [],
                    "estimator_raw_advantages": [],
                }
            elif accumulator[uid]["domain"] != domain:
                raise ValueError(
                    f"POISE prompt uid={uid!r} crossed domains while logging"
                )

            row = accumulator[uid]
            for name, values in field_values.items():
                row[name].append(float(values[row_index]))
            if member_values is not None and member_values[row_index] is not None:
                row.setdefault(
                    "estimator_member_value_predictions",
                    [],
                ).append(member_values[row_index])

    @staticmethod
    def _append_prompt_log_values(
        record: dict,
        row: dict,
        field_name: str,
    ) -> None:
        values = [float(value) for value in row[field_name]]
        record[field_name] = values
        record[f"{field_name}_mean"] = (
            float(np.mean(values)) if values else 0.0
        )
        record[f"{field_name}_std"] = (
            float(np.std(values)) if values else 0.0
        )

    @staticmethod
    def _append_prompt_log_error_summaries(record: dict) -> None:
        predictions = np.asarray(
            record["estimator_value_predictions"],
            dtype=np.float32,
        )
        targets = np.asarray(record["estimator_targets"], dtype=np.float32)
        estimator_diagnostics = regression_diagnostic_values(
            predictions=predictions,
            targets=targets,
        )
        record["estimator_online_mae"] = estimator_diagnostics["target_mae"]
        record["estimator_online_rmse"] = estimator_diagnostics["target_rmse"]
        record["estimator_online_constant_brier"] = (
            estimator_diagnostics["constant_brier"]
        )
        record["estimator_online_brier_skill"] = estimator_diagnostics[
            "brier_skill"
        ]
        record["estimator_online_brier_skill_defined"] = (
            estimator_diagnostics["brier_skill_defined"]
        )
        record["estimator_online_bias"] = estimator_diagnostics["target_bias"]

        baselines = np.asarray(
            record["estimator_cross_baselines"],
            dtype=np.float32,
        )
        rewards = np.asarray(record["rollout_rewards"], dtype=np.float32)
        baseline_diagnostics = regression_diagnostic_values(
            predictions=baselines,
            targets=rewards,
        )
        record["baseline_reward_mae"] = baseline_diagnostics["target_mae"]
        record["baseline_reward_rmse"] = baseline_diagnostics["target_rmse"]
        record["baseline_reward_constant_brier"] = baseline_diagnostics[
            "constant_brier"
        ]
        record["baseline_reward_brier_skill"] = baseline_diagnostics[
            "brier_skill"
        ]
        record["baseline_reward_brier_skill_defined"] = (
            baseline_diagnostics["brier_skill_defined"]
        )
        record["baseline_reward_bias"] = baseline_diagnostics["target_bias"]

    @staticmethod
    def _append_prompt_log_member_summaries(record: dict, row: dict) -> None:
        field_name = "estimator_member_value_predictions"
        if field_name not in row:
            return
        member_matrix = np.asarray(row[field_name], dtype=np.float32)
        if member_matrix.ndim != 2:
            raise ValueError(
                "Prompt reward log member predictions must have shape "
                f"[rollouts, members], got {member_matrix.shape}"
            )
        record[field_name] = [
            [float(value) for value in member_row]
            for member_row in member_matrix.tolist()
        ]
        record[f"{field_name}_mean"] = [
            float(value) for value in member_matrix.mean(axis=0).tolist()
        ]
        record["estimator_member_count"] = int(member_matrix.shape[1])
        targets = np.asarray(record["estimator_targets"], dtype=np.float32)
        member_summaries = []
        for member_index in range(member_matrix.shape[1]):
            diagnostics = regression_diagnostic_values(
                predictions=member_matrix[:, member_index],
                targets=targets,
            )
            member_summaries.append(
                {
                    "member_index": int(member_index),
                    "prediction_mean": diagnostics["prediction_mean"],
                    "prediction_std": diagnostics["prediction_std"],
                    "online_mae": diagnostics["target_mae"],
                    "online_rmse": diagnostics["target_rmse"],
                    "online_constant_brier": diagnostics[
                        "constant_brier"
                    ],
                    "online_brier_skill": diagnostics["brier_skill"],
                    "online_brier_skill_defined": diagnostics[
                        "brier_skill_defined"
                    ],
                    "online_bias": diagnostics["target_bias"],
                    "online_pearson": diagnostics["target_pearson"],
                }
            )
        record["estimator_member_summaries"] = member_summaries

    @staticmethod
    def _prompt_log_domain_summaries(records: list[dict]) -> dict[str, dict]:
        summaries = {}
        for domain in DOMAIN_NAMES:
            domain_records = [
                record for record in records if record["domain"] == domain
            ]
            if not domain_records:
                continue
            rewards = np.asarray(
                [
                    value
                    for record in domain_records
                    for value in record["rollout_rewards"]
                ],
                dtype=np.float32,
            )
            predictions = np.asarray(
                [
                    value
                    for record in domain_records
                    for value in record["estimator_value_predictions"]
                ],
                dtype=np.float32,
            )
            targets = np.asarray(
                [
                    value
                    for record in domain_records
                    for value in record["estimator_targets"]
                ],
                dtype=np.float32,
            )
            baselines = np.asarray(
                [
                    value
                    for record in domain_records
                    for value in record["estimator_cross_baselines"]
                ],
                dtype=np.float32,
            )
            estimator_diagnostics = regression_diagnostic_values(
                predictions=predictions,
                targets=targets,
            )
            baseline_diagnostics = regression_diagnostic_values(
                predictions=baselines,
                targets=rewards,
            )
            summary = {
                "prompt_count": int(len(domain_records)),
                "trajectory_count": int(rewards.size),
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "prediction_mean": estimator_diagnostics["prediction_mean"],
                "prediction_std": estimator_diagnostics["prediction_std"],
                "target_mean": estimator_diagnostics["target_mean"],
                "target_std": estimator_diagnostics["target_std"],
                "online_mae": estimator_diagnostics["target_mae"],
                "online_rmse": estimator_diagnostics["target_rmse"],
                "online_constant_brier": estimator_diagnostics[
                    "constant_brier"
                ],
                "online_brier_skill": estimator_diagnostics[
                    "brier_skill"
                ],
                "online_brier_skill_defined": estimator_diagnostics[
                    "brier_skill_defined"
                ],
                "online_bias": estimator_diagnostics["target_bias"],
                "online_pearson": estimator_diagnostics["target_pearson"],
                "baseline_reward_mae": baseline_diagnostics["target_mae"],
                "baseline_reward_rmse": baseline_diagnostics["target_rmse"],
                "baseline_reward_constant_brier": baseline_diagnostics[
                    "constant_brier"
                ],
                "baseline_reward_brier_skill": baseline_diagnostics[
                    "brier_skill"
                ],
                "baseline_reward_brier_skill_defined": baseline_diagnostics[
                    "brier_skill_defined"
                ],
                "baseline_reward_bias": baseline_diagnostics["target_bias"],
                "baseline_reward_pearson": baseline_diagnostics[
                    "target_pearson"
                ],
            }
            member_records = [
                record
                for record in domain_records
                if "estimator_member_value_predictions" in record
            ]
            if member_records:
                if len(member_records) != len(domain_records):
                    raise ValueError(
                        f"Only part of {domain} prompt logs contain member "
                        "predictions"
                    )
                member_matrix = np.concatenate(
                    [
                        np.asarray(
                            record["estimator_member_value_predictions"],
                            dtype=np.float32,
                        )
                        for record in member_records
                    ],
                    axis=0,
                )
                summary["estimator_member_count"] = int(
                    member_matrix.shape[1]
                )
                summary["member_summaries"] = []
                for member_index in range(member_matrix.shape[1]):
                    diagnostics = regression_diagnostic_values(
                        predictions=member_matrix[:, member_index],
                        targets=targets,
                    )
                    summary["member_summaries"].append(
                        {
                            "member_index": int(member_index),
                            "prediction_mean": diagnostics[
                                "prediction_mean"
                            ],
                            "prediction_std": diagnostics["prediction_std"],
                            "online_mae": diagnostics["target_mae"],
                            "online_rmse": diagnostics["target_rmse"],
                            "online_constant_brier": diagnostics[
                                "constant_brier"
                            ],
                            "online_brier_skill": diagnostics[
                                "brier_skill"
                            ],
                            "online_brier_skill_defined": diagnostics[
                                "brier_skill_defined"
                            ],
                            "online_bias": diagnostics["target_bias"],
                            "online_pearson": diagnostics["target_pearson"],
                        }
                    )
            else:
                summary["estimator_member_count"] = 0
            summaries[domain] = summary
        return summaries

    def _dump_prompt_reward_log(
        self,
        *,
        output_dir: str,
        accumulator: dict[str, dict],
        order: list[str],
        rollout_repeat: int,
        final_train_batch_rows: int,
    ) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        records = []
        for prompt_index, uid in enumerate(order):
            row = accumulator[uid]
            rewards = [float(value) for value in row["rollout_rewards"]]
            record = {
                "prompt_index": int(prompt_index),
                "uid": row["uid"],
                "domain": row["domain"],
                "data_source": row["data_source"],
                "prompt": row["prompt"],
                "raw_prompt": row["raw_prompt"],
                "reward_mean": (
                    float(np.mean(rewards)) if rewards else 0.0
                ),
                "num_rollouts": int(len(rewards)),
                "rollout_rewards": rewards,
            }
            for field_name in (
                "estimator_value_predictions",
                "estimator_cross_baselines",
                "estimator_targets",
                "estimator_raw_advantages",
            ):
                self._append_prompt_log_values(record, row, field_name)
            self._append_prompt_log_error_summaries(record)
            self._append_prompt_log_member_summaries(record, row)
            records.append(record)

        member_counts = {
            int(record["estimator_member_count"])
            for record in records
            if "estimator_member_count" in record
        }
        estimator_member_count = (
            int(next(iter(member_counts)))
            if len(member_counts) == 1
            and len(
                [
                    record
                    for record in records
                    if "estimator_member_count" in record
                ]
            )
            == len(records)
            else None
        )
        domain_summaries = self._prompt_log_domain_summaries(records)
        payload = {
            "schema_version": 7,
            "global_step": int(self.global_steps),
            "rollout_repeat": int(rollout_repeat),
            "final_train_batch_rows": int(final_train_batch_rows),
            "prompt_count": int(len(records)),
            "trajectory_count": int(
                sum(record["num_rollouts"] for record in records)
            ),
            "estimator_member_count": estimator_member_count,
            "estimator_member_counts_by_domain": {
                domain: summary["estimator_member_count"]
                for domain, summary in domain_summaries.items()
            },
            "domain_summaries": domain_summaries,
            "records": records,
        }

        filename = output_path / f"{self.global_steps}.json"
        temporary_filename = output_path / (
            f".{self.global_steps}.json.tmp-{os.getpid()}"
        )
        try:
            with temporary_filename.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_filename, filename)
        finally:
            temporary_filename.unlink(missing_ok=True)

        latest_path = output_path / self._PROMPT_REWARD_LATEST_FILENAME
        temporary_latest_path = output_path / (
            f".{self._PROMPT_REWARD_LATEST_FILENAME}.tmp-{os.getpid()}"
        )
        try:
            with temporary_latest_path.open("w", encoding="utf-8") as handle:
                handle.write(str(self.global_steps))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_latest_path, latest_path)
        finally:
            temporary_latest_path.unlink(missing_ok=True)
        return str(filename)

    def _resolve_resume_checkpoint_dir(self) -> str | None:
        if self.global_steps == 0:
            return None
        if self.config.trainer.resume_mode == "auto":
            return find_latest_ckpt_path(self._default_local_checkpoint_root())
        if self.config.trainer.resume_mode == "resume_path":
            checkpoint_dir = self.config.trainer.resume_from_path
            if checkpoint_dir and not os.path.isabs(checkpoint_dir):
                checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
            return checkpoint_dir
        return None

    def _checkpoint_dir(self) -> str:
        return os.path.join(
            self._default_local_checkpoint_root(),
            f"global_step_{self.global_steps}",
        )

    def _save_checkpoint(self):
        if self._scratch_bootstrap_enabled():
            self._save_phase_state(self._checkpoint_dir())
        if hasattr(self, "_poise_bank"):
            self._poise_bank.save(self._checkpoint_dir())
            print(f"[POISE] Saved domain estimator state to {self._checkpoint_dir()}")
        # The base implementation writes latest_checkpointed_iteration.txt last.
        # Saving POISE state first prevents that pointer from selecting a checkpoint
        # whose estimator state was interrupted or never written.
        super()._save_checkpoint()

    def _restore_estimator_checkpoint(self) -> None:
        resume_dir = self._resolve_resume_checkpoint_dir()
        if resume_dir is None:
            if self.global_steps != 0:
                raise RuntimeError(
                    "Actor state resumed, but its POISE checkpoint directory "
                    "could not be resolved."
                )
            return
        if not self._poise_bank.load(resume_dir):
            raise FileNotFoundError(
                "Actor state resumed without the required POISE estimator state: "
                f"{os.path.join(resume_dir, STATE_FILENAME)}"
            )
        print(f"[POISE] Restored domain estimator state from {resume_dir}")

    def _save_phase_state(self, checkpoint_dir: str) -> None:
        output_dir = Path(checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "phase": self._poise_phase,
            "bootstrap_steps": int(self.config.trainer.poise.bootstrap.steps),
            "bootstrap_mode": "rloo",
        }
        if self._poise_phase == "bootstrap":
            payload["buffer"] = self._bootstrap_buffer.to_payload()
        state_path = output_dir / self._PHASE_STATE_FILENAME
        temporary_path = output_dir / (
            f".{self._PHASE_STATE_FILENAME}.tmp-{os.getpid()}"
        )
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, state_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _restore_scratch_checkpoint(self) -> None:
        resume_dir = self._resolve_resume_checkpoint_dir()
        if resume_dir is None:
            if self.global_steps != 0:
                raise RuntimeError(
                    "Actor state resumed without a resolvable POISE phase checkpoint"
                )
            return
        state_path = Path(resume_dir) / self._PHASE_STATE_FILENAME
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Scratch POISE phase state is missing: {state_path}"
            )
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported POISE phase state: {state_path}")
        configured_steps = int(self.config.trainer.poise.bootstrap.steps)
        if int(payload["bootstrap_steps"]) != configured_steps:
            raise ValueError(
                "POISE bootstrap steps changed across resume: "
                f"{payload['bootstrap_steps']} != {configured_steps}"
            )
        # Retain the marker for compatibility with existing RLOO checkpoints.
        saved_mode = str(payload.get("bootstrap_mode", "")).lower()
        if saved_mode != "rloo":
            raise ValueError(
                "POISE requires an RLOO bootstrap checkpoint; "
                f"got bootstrap_mode={saved_mode!r}"
            )
        phase = str(payload["phase"])
        if phase == "bootstrap":
            self._bootstrap_buffer = RecentDomainBuffer.from_payload(
                payload["buffer"],
                expected_buffer_max_rows=self._poise_runtime.buffer_max_rows,
            )
            self._poise_phase = "bootstrap"
            print(f"[POISE] Restored scratch bootstrap state from {state_path}")
            return
        if phase != "poise":
            raise ValueError(f"Unknown POISE checkpoint phase: {phase!r}")
        self._poise_bank = DomainEstimatorBank.from_checkpoint(
            resume_dir,
            **self._estimator_runtime_kwargs(),
        )
        self._set_estimator_runtime(self._poise_bank)
        self._poise_phase = "poise"
        print(f"[POISE] Restored post-bootstrap estimator state from {resume_dir}")

    def fit(self):
        self._validate_poise_config()

        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        try:
            result = self._fit_with_tracking(logger)
        except BaseException:
            try:
                logger.close(exit_code=1)
            except Exception as close_error:
                print(f"Tracking shutdown after training failure also failed: {close_error}")
            raise
        else:
            logger.close(exit_code=0)
            return result

    def _fit_with_tracking(self, logger):
        """Run POISE while fit() owns and closes the experiment logger."""

        self.global_steps = 0
        self._load_checkpoint()
        if self._scratch_bootstrap_enabled():
            self._initialize_scratch_bootstrap()
            self._restore_scratch_checkpoint()
        else:
            self._poise_bank = self._initialize_estimator_bank()
            self._poise_phase = "poise"
            self._restore_estimator_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            if (
                self._poise_phase == "bootstrap"
                and bool(self.config.trainer.poise.estimator.eval_on_val)
            ):
                val_metrics = (
                    self._validate_initial_estimator_for_scratch_bootstrap()
                )
            else:
                val_metrics = self._validate()
            assert val_metrics
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(
                data=self._compact_wandb_metrics(val_metrics),
                step=self.global_steps,
            )
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )
        self.global_steps += 1
        last_val_metrics = None
        timing_raw = defaultdict(float)

        prompt_batch_size = int(self.config.data.train_batch_size)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        world_size = int(self.resource_pool_manager.get_n_gpus())
        prompt_divisor = world_size // math.gcd(world_size, rollout_n)
        prompt_reward_log_dir = self._resolve_prompt_reward_log_dir()

        if prompt_batch_size <= 0 or prompt_batch_size % prompt_divisor:
            raise ValueError(
                "data.train_batch_size must be positive and make "
                "train_batch_size * rollout.n divisible by the GPU count"
            )
        prompt_buffer = None

        num_batches_per_epoch = len(self.train_dataloader)
        for epoch in range(self.config.trainer.total_epochs):
            for batch_index, batch_dict in enumerate(self.train_dataloader):
                incoming = DataProto.from_single_dict(batch_dict)
                incoming = self._normalize_non_tensor_rows(incoming)
                incoming.non_tensor_batch["uid"] = self._stable_prompt_uids(incoming)
                bootstrap_active = self._poise_phase == "bootstrap"
                existing_uids = (
                    {
                        str(uid)
                        for uid in prompt_buffer.non_tensor_batch["uid"]
                    }
                    if prompt_buffer is not None
                    else set()
                )
                unique_indices = self._unique_prompt_indices(incoming, existing_uids)
                if unique_indices:
                    incoming = incoming[unique_indices]
                    prompt_buffer = (
                        incoming
                        if prompt_buffer is None
                        else DataProto.concat([prompt_buffer, incoming])
                    )
                if prompt_buffer is None:
                    continue

                if len(prompt_buffer.batch) < prompt_batch_size:
                    continue

                new_batch = prompt_buffer[:prompt_batch_size]
                remainder = prompt_buffer[prompt_batch_size:]
                prompt_buffer = remainder if len(remainder.batch) else None
                prompt_reward_log_accumulator: dict[str, dict] = {}
                prompt_reward_log_order: list[str] = []
                metrics: dict[str, float] = {}
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )

                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(
                            role="e2e", profile_step=self.global_steps
                        )
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    if "multi_modal_data" in new_batch.non_tensor_batch:
                        gen_batch = new_batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=[
                                "raw_prompt_ids",
                                "multi_modal_data",
                            ],
                        )
                    else:
                        gen_batch = new_batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids"],
                        )
                    gen_batch = gen_batch.repeat(
                        repeat_times=rollout_n, interleave=True
                    )

                    with marked_timer("gen", timing_raw, "red"):
                        gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_output.meta_info["timing"])
                        gen_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            baseline_batch = deepcopy(gen_batch)
                            baseline_batch.meta_info["do_sample"] = False
                            baseline_output = self.actor_rollout_wg.generate_sequences(
                                baseline_batch
                            )
                            new_batch = new_batch.union(baseline_output)
                            reward_baseline = self.reward_fn(new_batch).sum(dim=-1)
                            new_batch.pop(
                                batch_keys=list(baseline_output.batch.keys())
                            )
                            new_batch.batch["reward_baselines"] = reward_baseline

                    new_batch = new_batch.repeat(
                        repeat_times=rollout_n, interleave=True
                    )
                    batch = new_batch.union(gen_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm:
                            rm_scores = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(rm_scores)
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos = reward_result.get(
                                "reward_extra_info", {}
                            )
                        except Exception as exc:
                            print(f"Error in reward_fn: {exc}")
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos = {}
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos:
                            batch.non_tensor_batch.update(
                                {
                                    key: np.asarray(value)
                                    for key, value in reward_extra_infos.items()
                                }
                            )
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = reward_tensor

                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        batch.meta_info["estimator_hidden_capture"] = dict(
                            self._poise_capture_spec
                        )
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch.meta_info.pop("estimator_hidden_capture", None)
                        entropies = old_log_prob.batch["entropys"]
                        entropy_agg = agg_loss(
                            loss_mat=entropies,
                            loss_mask=batch.batch["response_mask"],
                            loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                        )
                        metrics["actor/entropy"] = float(entropy_agg.detach().item())
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    with marked_timer("adv", timing_raw, "brown"):
                        reward_sums = reward_tensor.sum(dim=-1).to(torch.float32)
                        if bootstrap_active:
                            estimator_rows, poise_metrics = (
                                self._compute_bootstrap_rloo_advantages(
                                    batch=batch,
                                    reward_sums=reward_sums,
                                    entropies=entropies,
                                )
                            )
                            self._bootstrap_buffer.append(estimator_rows)
                        else:
                            estimator_rows, poise_metrics = (
                                self._compute_poise_advantages(
                                    batch=batch,
                                    reward_sums=reward_sums,
                                    entropies=entropies,
                                )
                            )
                        metrics.update(poise_metrics)
                        if not bootstrap_active:
                            self._accumulate_prompt_reward_log_rows(
                                accumulator=prompt_reward_log_accumulator,
                                order=prompt_reward_log_order,
                                batch=batch,
                                reward_sums=reward_sums,
                            )
                        # Feature rows already own CPU copies of these tensors.
                        # Release hidden states before the actor update;
                        # old log-probabilities remain in the batch.
                        batch.batch.pop("estimator_prompt_hidden")
                        batch.batch.pop("estimator_response_hidden")

                    rloo_batch_order = (
                        bootstrap_active
                        and self._bootstrap_uses_rloo_batch_order()
                    )
                    final_indices = self._actor_batch_indices(
                        batch,
                        prompt_count=prompt_batch_size,
                        group_size=rollout_n,
                        preserve_order=rloo_batch_order,
                    )
                    batch = batch[final_indices]
                    response_mask = batch.batch["response_mask"].to(torch.float32)
                    sequence_advantages = batch.batch[
                        "poise_raw_advantages"
                    ].to(torch.float32)
                    batch.batch["advantages"] = (
                        sequence_advantages.unsqueeze(-1) * response_mask
                    )
                    batch.batch["returns"] = batch.batch["advantages"]
                    # rloo_batch_order keeps the balancing pass that already
                    # ran right after generation, so this second pass would
                    # only reorder an already balanced batch.
                    if self.config.trainer.balance_batch and not rloo_batch_order:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    final_domains = [
                        domain_from_data_source(source)
                        for source in batch.non_tensor_batch["data_source"]
                    ]
                    domain_counts = self._poise_runtime.domain_counts(
                        final_domains
                    )
                    for domain, trajectory_count in domain_counts.items():
                        metrics[
                            f"poise/final_batch/{domain}_prompts"
                        ] = float(trajectory_count / rollout_n)
                    if bootstrap_active:
                        metrics.update(self._bootstrap_buffer_metrics())
                    else:
                        metrics.update(
                            self._estimator_online_metrics(estimator_rows)
                        )
                        metrics.update(
                            self._poise_bank.update(estimator_rows)
                        )

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = (
                                self.ref_policy_wg.compute_ref_log_prob(batch)
                            )
                            batch = batch.union(ref_log_prob)
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                        with marked_timer(
                            "update_critic", timing_raw, "pink"
                        ):
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics.update(
                            reduce_metrics(
                                critic_output.meta_info["metrics"]
                            )
                        )
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer(
                            "update_actor", timing_raw, "red"
                        ):
                            actor_output = self.actor_rollout_wg.update_actor(
                                batch
                            )
                        metrics.update(
                            reduce_metrics(actor_output.meta_info["metrics"])
                        )

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                if not bootstrap_active:
                    with marked_timer(
                        "dump_prompt_reward_log",
                        timing_raw,
                        "green",
                    ):
                        prompt_log_filename = self._dump_prompt_reward_log(
                            output_dir=prompt_reward_log_dir,
                            accumulator=prompt_reward_log_accumulator,
                            order=prompt_reward_log_order,
                            rollout_repeat=rollout_n,
                            final_train_batch_rows=len(batch.batch),
                        )
                    metrics["poise/prompt_reward_log/prompt_count"] = float(
                        len(prompt_reward_log_order)
                    )
                    metrics["poise/prompt_reward_log/trajectory_count"] = float(
                        sum(
                            len(
                                prompt_reward_log_accumulator[uid][
                                    "rollout_rewards"
                                ]
                            )
                            for uid in prompt_reward_log_order
                        )
                    )
                    print(
                        f"[POISE] Saved prompt reward log to {prompt_log_filename}"
                    )

                bootstrap_transitioned = False
                if bootstrap_active and self.global_steps >= int(
                    self.config.trainer.poise.bootstrap.steps
                ):
                    with marked_timer(
                        "fit_bootstrap_estimators",
                        timing_raw,
                        "green",
                    ):
                        metrics.update(self._fit_bootstrap_estimators())
                    bootstrap_transitioned = True

                # Estimator snapshots have their own cadence and deliberately
                # run before validation, so test_freq/validation failures cannot
                # suppress an otherwise due adaptive-estimator save.
                if self._should_save_estimator_update():
                    with marked_timer(
                        "save_estimator_update",
                        timing_raw,
                        "green",
                    ):
                        self._save_estimator_update_snapshot()
                    metrics["poise/estimator/snapshot_saved"] = 1.0
                else:
                    metrics["poise/estimator/snapshot_saved"] = 0.0

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (
                        is_last_step
                        or self.global_steps % self.config.trainer.test_freq == 0
                    )
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if bootstrap_transitioned or (
                    self.config.trainer.save_freq > 0
                    and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    )
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_data_epoch_metrics(
                        epoch=epoch,
                        batch_index=batch_index,
                        num_batches_per_epoch=num_batches_per_epoch,
                    )
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                metrics["training/global_step"] = self.global_steps
                logger.log(
                    data=self._compact_wandb_metrics(metrics),
                    step=self.global_steps,
                )
                timing_raw = defaultdict(float)

                progress_bar.update(1)
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                self.global_steps += 1

        progress_bar.close()
