# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from typing import Any

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.adam_variance import (
    adam_variance_enabled,
    append_adam_variance_jsonl,
    measure_adam_variance_proxy,
)
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self._last_hidden_capture: dict[str, torch.Tensor] | None = None
        self._capture_layer_cache: dict[int, nn.Module] = {}

    def _resolve_transformer_layers(self):
        base_model = self.actor_module
        for _ in range(4):
            next_model = getattr(base_model, "module", None)
            if next_model is None or next_model is base_model:
                break
            base_model = next_model

        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            return base_model.model.layers
        if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            return base_model.transformer.h
        if hasattr(base_model, "gpt_neox") and hasattr(base_model.gpt_neox, "layers"):
            return base_model.gpt_neox.layers
        raise ValueError(
            "Could not resolve transformer layers for POISE hidden capture. "
            "Expected model.layers, transformer.h, or gpt_neox.layers."
        )

    def _resolve_capture_layer(self, layer_index: int) -> nn.Module:
        if layer_index not in self._capture_layer_cache:
            layers = self._resolve_transformer_layers()
            if layer_index < 0 or layer_index >= len(layers):
                raise ValueError(
                    f"POISE capture layer {layer_index} is invalid for {len(layers)} layers"
                )
            self._capture_layer_cache[layer_index] = layers[layer_index]
        return self._capture_layer_cache[layer_index]

    @staticmethod
    def _pool_last_n_tokens(
        hidden_tokens: torch.Tensor, n: int, hidden_dim: int
    ) -> torch.Tensor:
        if hidden_tokens.numel() == 0:
            return torch.zeros(
                hidden_dim, device=hidden_tokens.device, dtype=torch.float32
            )
        n = max(1, min(int(n), hidden_tokens.size(0)))
        return hidden_tokens[-n:].mean(dim=0).to(torch.float32)

    @staticmethod
    def _find_last_subsequence(ids: list[int], subsequence: list[int]) -> int | None:
        if not subsequence or len(ids) < len(subsequence):
            return None
        for start in range(len(ids) - len(subsequence), -1, -1):
            if ids[start : start + len(subsequence)] == subsequence:
                return start + len(subsequence) - 1
        return None

    def _build_hidden_capture(
        self,
        *,
        prompt_full_hidden: torch.Tensor,
        response_full_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor | None,
        response_length: int,
        prompt_pool_n: int,
        response_pool_n: int,
        response_ids: torch.Tensor | None,
        think_end_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        batch_size, sequence_length, prompt_hidden_dim = prompt_full_hidden.shape
        response_shape = tuple(response_full_hidden.shape)
        if response_shape[:2] != (batch_size, sequence_length):
            raise ValueError(
                "POISE prompt/response hidden states must share batch and sequence "
                f"dimensions, got {tuple(prompt_full_hidden.shape)} and "
                f"{response_shape}"
            )
        response_hidden_dim = response_shape[2]
        prompt_length = sequence_length - response_length
        if prompt_length <= 0:
            raise ValueError(f"Invalid POISE prompt length: {prompt_length}")

        prompt_rows = []
        response_rows = []
        for row_index in range(batch_size):
            prompt_tokens = prompt_full_hidden[row_index, :prompt_length]
            prompt_tokens = prompt_tokens[
                attention_mask[row_index, :prompt_length].bool()
            ]
            prompt_rows.append(
                self._pool_last_n_tokens(
                    prompt_tokens, prompt_pool_n, prompt_hidden_dim
                )
            )

            response_hidden = response_full_hidden[row_index, prompt_length:]
            valid_response = (
                response_mask[row_index].bool()
                if response_mask is not None
                else attention_mask[row_index, prompt_length:].bool()
            )
            response_tokens = response_hidden[valid_response]
            if response_ids is not None and think_end_token_ids:
                valid_ids = response_ids[row_index][valid_response].tolist()
                think_end = self._find_last_subsequence(
                    valid_ids, think_end_token_ids
                )
                if think_end is not None:
                    response_tokens = response_tokens[: think_end + 1]
            response_rows.append(
                self._pool_last_n_tokens(
                    response_tokens, response_pool_n, response_hidden_dim
                )
            )

        return {
            "prompt_hidden": torch.stack(prompt_rows),
            "response_hidden": torch.stack(response_rows),
        }

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        hidden_capture_spec: dict[str, Any] | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
    ]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            hidden capture: pooled prompt and response states, or None
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            hidden_capture = None
            captured_layer_outputs: dict[int, torch.Tensor] = {}
            layer_hook_handles = []
            if hidden_capture_spec is not None:
                capture_layer_indices = sorted(
                    {
                        int(hidden_capture_spec["prompt_layer_index"]),
                        int(hidden_capture_spec["response_layer_index"]),
                    }
                )

                def make_capture_hook(layer_index: int):
                    def capture_hidden(
                        _module: nn.Module, _inputs: tuple[Any, ...], output: Any
                    ) -> None:
                        captured_layer_outputs[layer_index] = (
                            output[0] if isinstance(output, tuple) else output
                        )

                    return capture_hidden

                for layer_index in capture_layer_indices:
                    capture_layer = self._resolve_capture_layer(layer_index)
                    layer_hook_handles.append(
                        capture_layer.register_forward_hook(
                            make_capture_hook(layer_index)
                        )
                    )

            def remove_layer_hooks() -> None:
                while layer_hook_handles:
                    layer_hook_handles.pop().remove()

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                try:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating
                finally:
                    remove_layer_hooks()

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                captured_hidden_rmpad: dict[int, torch.Tensor] = {}
                if hidden_capture_spec is not None:
                    for layer_index in capture_layer_indices:
                        captured_hidden = captured_layer_outputs.get(layer_index)
                        if captured_hidden is None:
                            raise RuntimeError(
                                "POISE failed to capture hidden states from layer "
                                f"{layer_index} in the remove-padding path"
                            )
                        if captured_hidden.dim() == 3:
                            captured_hidden = captured_hidden.squeeze(0)
                        if captured_hidden.dim() != 2:
                            raise ValueError(
                                "Expected POISE hidden states [tokens, hidden_dim] "
                                f"from layer {layer_index}, got "
                                f"{tuple(captured_hidden.shape)}"
                            )
                        if self.use_ulysses_sp:
                            captured_hidden = gather_outputs_and_unpad(
                                captured_hidden,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                        captured_hidden_rmpad[layer_index] = captured_hidden
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if captured_hidden_rmpad:
                    full_hidden_by_layer = {
                        layer_index: pad_input(
                            hidden_states=captured_hidden,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        for layer_index, captured_hidden in captured_hidden_rmpad.items()
                    }
                    hidden_capture = self._build_hidden_capture(
                        prompt_full_hidden=full_hidden_by_layer[
                            int(hidden_capture_spec["prompt_layer_index"])
                        ],
                        response_full_hidden=full_hidden_by_layer[
                            int(hidden_capture_spec["response_layer_index"])
                        ],
                        attention_mask=attention_mask,
                        response_mask=micro_batch.get("response_mask"),
                        response_length=response_length,
                        prompt_pool_n=int(hidden_capture_spec["prompt_pool_n"]),
                        response_pool_n=int(hidden_capture_spec["response_pool_n"]),
                        response_ids=micro_batch.get("responses"),
                        think_end_token_ids=hidden_capture_spec.get(
                            "think_end_token_ids", []
                        ),
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                try:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating
                finally:
                    remove_layer_hooks()

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if hidden_capture_spec is not None:
                    for layer_index in capture_layer_indices:
                        captured_hidden = captured_layer_outputs.get(layer_index)
                        if captured_hidden is None:
                            raise RuntimeError(
                                "POISE failed to capture hidden states from layer "
                                f"{layer_index} in the dense path"
                            )
                        if captured_hidden.dim() != 3:
                            raise ValueError(
                                "Expected POISE hidden states [batch, sequence, "
                                f"hidden_dim] from layer {layer_index}, got "
                                f"{tuple(captured_hidden.shape)}"
                            )
                    hidden_capture = self._build_hidden_capture(
                        prompt_full_hidden=captured_layer_outputs[
                            int(hidden_capture_spec["prompt_layer_index"])
                        ],
                        response_full_hidden=captured_layer_outputs[
                            int(hidden_capture_spec["response_layer_index"])
                        ],
                        attention_mask=attention_mask,
                        response_mask=micro_batch.get("response_mask"),
                        response_length=response_length,
                        prompt_pool_n=int(hidden_capture_spec["prompt_pool_n"]),
                        response_pool_n=int(hidden_capture_spec["response_pool_n"]),
                        response_ids=micro_batch.get("responses"),
                        think_end_token_ids=hidden_capture_spec.get(
                            "think_end_token_ids", []
                        ),
                    )

            remove_layer_hooks()
            return entropy, log_probs, hidden_capture

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        hidden_capture_spec = data.meta_info.get("estimator_hidden_capture")
        if hidden_capture_spec is not None:
            legacy_layer_index = hidden_capture_spec.get("layer_index")
            prompt_layer_index = hidden_capture_spec.get(
                "prompt_layer_index", legacy_layer_index
            )
            response_layer_index = hidden_capture_spec.get(
                "response_layer_index", legacy_layer_index
            )
            if prompt_layer_index is None or response_layer_index is None:
                raise ValueError(
                    "POISE hidden capture requires prompt_layer_index and "
                    "response_layer_index (or legacy layer_index)"
                )
            hidden_capture_spec = {
                "prompt_layer_index": int(prompt_layer_index),
                "response_layer_index": int(response_layer_index),
                "prompt_pool_n": int(hidden_capture_spec["prompt_pool_n"]),
                "response_pool_n": int(hidden_capture_spec["response_pool_n"]),
                "think_end_token_ids": [
                    int(token_id)
                    for token_id in hidden_capture_spec.get(
                        "think_end_token_ids", []
                    )
                ],
            }
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if hidden_capture_spec is not None and "response_mask" in data.batch:
            select_keys.append("response_mask")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        prompt_hidden_lst = []
        response_hidden_lst = []
        self._last_hidden_capture = None
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, hidden_capture = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    hidden_capture_spec=hidden_capture_spec,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if hidden_capture is not None:
                prompt_hidden_lst.append(hidden_capture["prompt_hidden"])
                response_hidden_lst.append(hidden_capture["response_hidden"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        prompt_hidden = None
        response_hidden = None
        if prompt_hidden_lst:
            prompt_hidden = torch.concat(prompt_hidden_lst, dim=0)
            response_hidden = torch.concat(response_hidden_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if prompt_hidden is not None:
                prompt_hidden = restore_dynamic_batch(
                    prompt_hidden, batch_idx_list
                )
                response_hidden = restore_dynamic_batch(
                    response_hidden, batch_idx_list
                )

        if prompt_hidden is not None:
            self._last_hidden_capture = {
                "prompt_hidden": prompt_hidden.to(torch.float32),
                "response_hidden": response_hidden.to(torch.float32),
            }
        return log_probs, entropys

    def pop_hidden_capture(self) -> dict[str, torch.Tensor] | None:
        hidden_capture = self._last_hidden_capture
        self._last_hidden_capture = None
        return hidden_capture

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if adam_variance_enabled():
            proxy = measure_adam_variance_proxy(self.actor_optimizer)
            append_to_dict(
                metrics,
                {
                    "actor/adam_grad_variance_proxy_sum": proxy["sum"],
                    "actor/adam_grad_variance_proxy_mean": proxy["mean"],
                    "actor/adam_grad_variance_proxy_negative_fraction": proxy[
                        "negative_fraction"
                    ],
                    "actor/adam_variance_numel": proxy["numel"],
                    "actor/adam_optimizer_step": proxy["optimizer_step"],
                },
            )
            append_adam_variance_jsonl(proxy)
        return metrics
