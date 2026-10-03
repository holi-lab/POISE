# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Prompt-only binary critic for Sequence-Level PPO.

This is deliberately separate from :mod:`verl.workers.critic.dp_critic` so the
token-value PPO/GAE implementation retains its original forward pass and loss.
"""

import logging
import os

import torch
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelSequencePPOCritic(BasePPOCritic):
    """Estimate ``P(success | prompt)`` from one prompt-level value logit."""

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        if self.config.model.get("use_remove_padding", False):
            raise ValueError("The Sequence-Level PPO critic currently requires critic.model.use_remove_padding=False")

    @staticmethod
    def _prompt_only_data(data: DataProto, include_returns: bool = False) -> DataProto:
        """Drop response tokens while preserving sample order and BCE targets."""
        response_length = data.batch["responses"].size(-1)
        position_ids = data.batch["position_ids"]
        if position_ids.ndim == 3:
            prompt_position_ids = position_ids[:, :, :-response_length]
        else:
            prompt_position_ids = position_ids[:, :-response_length]

        tensors = {
            "input_ids": data.batch["input_ids"][:, :-response_length],
            "attention_mask": data.batch["attention_mask"][:, :-response_length],
            "position_ids": prompt_position_ids,
        }
        if include_returns:
            tensors["returns"] = data.batch["returns"]

        non_tensors = {}
        if "multi_modal_inputs" in data.non_tensor_batch:
            non_tensors["multi_modal_inputs"] = data.non_tensor_batch["multi_modal_inputs"]
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    def _forward_prompt_value(self, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([item[key] for item in micro_batch["multi_modal_inputs"]], dim=0)

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        if position_ids.ndim == 3:
            position_ids = position_ids.transpose(0, 1)

        if torch.any(attention_mask.sum(dim=-1) == 0):
            raise ValueError("Sequence-Level PPO received an empty prompt")

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.critic_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            if hasattr(self.critic_module, "v_head"):
                value_sequence = output[2]
            else:
                value_sequence = output.logits

            if value_sequence.ndim == 3 and value_sequence.size(-1) == 1:
                value_sequence = value_sequence.squeeze(-1)
            if value_sequence.ndim != 2:
                raise ValueError(f"Expected critic values with shape [batch, sequence], got {value_sequence.shape}")

            token_positions = torch.arange(attention_mask.size(-1), device=attention_mask.device)
            last_prompt_indices = (attention_mask.to(torch.long) * token_positions).argmax(dim=-1)
            return value_sequence.gather(1, last_prompt_indices.unsqueeze(-1)).squeeze(-1)

    def _optimizer_step(self):
        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        if torch.isfinite(grad_norm):
            self.critic_optimizer.step()
        else:
            logger.warning("Skipping sequence critic update because grad_norm is not finite: %s", grad_norm)
            self.critic_optimizer.zero_grad()
        return grad_norm

    @GPUMemoryLogger(role="sequence dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        prompt_data = self._prompt_only_data(data)
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(prompt_data, max_token_len=max_token_len)
        else:
            micro_batches = prompt_data.split(data.meta_info["micro_batch_size"])

        value_parts = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                value_parts.append(self._forward_prompt_value({**micro_batch.batch, **micro_batch.non_tensor_batch}))
        values = torch.cat(value_parts, dim=0)
        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)
        return values

    @GPUMemoryLogger(role="sequence dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        self.critic_module.train()
        metrics = {}
        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "returns"]
        non_tensor_select_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        for _ in range(self.config.ppo_epochs):
            for mini_batch in data.split(self.config.ppo_mini_batch_size):
                prompt_data = self._prompt_only_data(mini_batch, include_returns=True)
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(prompt_data, max_token_len=max_token_len)
                else:
                    micro_batches = prompt_data.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()
                mini_batch_size = len(prompt_data)
                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    targets = model_inputs.pop("returns").float()
                    value_logits = self._forward_prompt_value(model_inputs)
                    loss_per_sample = self.bce_loss(value_logits, targets)
                    value_loss = loss_per_sample.mean()
                    loss_scale = len(micro_batch) / mini_batch_size
                    (value_loss * loss_scale).backward()

                    append_to_dict(
                        metrics,
                        {
                            "critic/vf_loss": value_loss.detach().item(),
                            "critic/vf_clipfrac": 0.0,
                            "critic/vpred_prob_mean": torch.sigmoid(value_logits).mean().detach().item(),
                        },
                    )

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})

        self.critic_optimizer.zero_grad()
        return metrics
