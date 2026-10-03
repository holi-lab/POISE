"""Adam moment-state proxy for the variance of policy gradients.

This intentionally measures the same quantity as the Dolci Adam-variance
baseline: after an optimizer update, bias-correct Adam's exponential first and
second moments and sum ``v_hat - m_hat**2`` over every optimizer-state element.
It is an inexpensive, history-weighted proxy; it is not an unbiased estimate of
the variance of the gradient from the current training batch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


def adam_variance_enabled() -> bool:
    """Return whether actor-side Adam variance instrumentation is enabled."""

    return os.environ.get("VERL_ENABLE_ADAM_VARIANCE", "0") == "1"


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def _optimizer_step_number(state: dict[str, Any]) -> int:
    step = state.get("step", 0)
    if isinstance(step, torch.Tensor):
        step = _local_tensor(step)
        return int(step.detach().cpu().item())
    return int(step)


@torch.no_grad()
def measure_adam_variance_proxy(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float | int]:
    """Measure the global ``sum(v_hat - m_hat**2)`` from Adam state.

    FSDP optimizer states are sharded, so every rank computes a local partial
    sum and the result is all-reduced. The returned sum, mean, element count,
    negative fraction, and optimizer step are consequently global values.
    """

    totals: torch.Tensor | None = None
    local_steps: list[int] = []
    state_tensor_count = 0
    chunk_size = max(
        1,
        int(
            os.environ.get(
                "VERL_ADAM_VARIANCE_CHUNK_SIZE",
                os.environ.get(
                    "DOLCI_ADAM_PROXY_CHUNK_SIZE", str(8 * 1024 * 1024)
                ),
            )
        ),
    )

    for group in optimizer.param_groups:
        if "betas" not in group:
            raise RuntimeError(
                "Adam variance logging requires an optimizer with beta moments"
            )
        beta1, beta2 = (float(value) for value in group["betas"])
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
                continue
            step = _optimizer_step_number(state)
            if step <= 0:
                continue

            exp_avg = _local_tensor(state["exp_avg"]).reshape(-1)
            exp_avg_sq = _local_tensor(state["exp_avg_sq"]).reshape(-1)
            if exp_avg.numel() != exp_avg_sq.numel():
                raise RuntimeError("Adam first- and second-moment states have different sizes")
            if totals is None:
                # Keeping the reduction tensor beside the optimizer state makes
                # this work with NCCL-sharded CUDA state and CPU-only tests.
                totals = torch.zeros(3, dtype=torch.float64, device=exp_avg.device)

            local_steps.append(step)
            state_tensor_count += 1
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            for start in range(0, exp_avg.numel(), chunk_size):
                stop = min(start + chunk_size, exp_avg.numel())
                m_hat = exp_avg[start:stop].to(dtype=torch.float32) / bias_correction1
                v_hat = exp_avg_sq[start:stop].to(dtype=torch.float32) / bias_correction2
                variance = v_hat - m_hat.square()
                totals[0] += variance.sum(dtype=torch.float64)
                totals[1] += variance.numel()
                totals[2] += (variance < 0).sum(dtype=torch.float64)

    if totals is None or not local_steps:
        raise RuntimeError("Adam has no initialized moment states after the policy update")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        local_step_bounds = torch.tensor(
            [min(local_steps), max(local_steps)],
            dtype=torch.int64,
            device=totals.device,
        )
        global_step_min = local_step_bounds[:1].clone()
        global_step_max = local_step_bounds[1:].clone()
        torch.distributed.all_reduce(global_step_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(global_step_max, op=torch.distributed.ReduceOp.MAX)
        step_min, step_max = int(global_step_min.item()), int(global_step_max.item())
    else:
        step_min, step_max = min(local_steps), max(local_steps)

    if step_min != step_max:
        raise RuntimeError(f"Adam state steps disagree: min={step_min}, max={step_max}")
    proxy_sum, numel, negative_count = (float(value) for value in totals.cpu().tolist())
    if numel <= 0:
        raise RuntimeError("Adam variance proxy covered zero elements")
    return {
        "optimizer_step": step_min,
        "sum": proxy_sum,
        "mean": proxy_sum / numel,
        "numel": int(numel),
        "negative_fraction": negative_count / numel,
        "local_state_tensor_count": state_tensor_count,
    }


def _is_log_rank() -> bool:
    return not (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ) or torch.distributed.get_rank() == 0


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in (None, "") else int(value)


def append_adam_variance_jsonl(proxy: dict[str, float | int]) -> None:
    """Append one proxy observation on distributed rank zero."""

    log_path = os.environ.get("VERL_ADAM_VARIANCE_LOG_PATH")
    if not log_path or not _is_log_rank():
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "run_name": os.environ.get("VERL_ADAM_VARIANCE_RUN_NAME"),
        "algorithm": os.environ.get("VERL_ADAM_VARIANCE_ALGORITHM"),
        "model": os.environ.get("VERL_ADAM_VARIANCE_MODEL"),
        "rollout_n": _optional_int_env("VERL_ADAM_VARIANCE_ROLLOUT_N"),
        "train_prompt_batch_size": _optional_int_env(
            "VERL_ADAM_VARIANCE_TRAIN_PROMPT_BATCH_SIZE"
        ),
        "train_trajectories": _optional_int_env(
            "VERL_ADAM_VARIANCE_TRAIN_TRAJECTORIES"
        ),
        **proxy,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
