# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""POISE training entry point for the bundled Ray/FSDP runtime."""

import os
import random
import socket
from pathlib import Path

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from .trainer import RayPOISETrainer


# Resolve packaged feature files independently of the caller's working directory.
OmegaConf.register_new_resolver(
    "poise_config", lambda name: str(Path(__file__).parent / "config" / name), replace=True
)


_POISE_VALIDATION_FILE_ENV = {
    "math__aime_repeated_8x_240.parquet": "POISE_AIME_VAL_FILE",
    "math__amc_repeated_4x_332.parquet": "POISE_AMC_VAL_FILE",
}


def seed_poise_task_runner(config) -> int:
    """Match the shared RLOO driver RNG initialization exactly."""
    seed = int(config.data.get("seed", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@hydra.main(config_path="config", config_name="poise_fsdp", version_base=None)
def main(config):
    run_ppo(config)


def apply_poise_validation_rollout_config(config) -> None:
    """Apply POISE-only sampling when estimator diagnostics are enabled."""
    if not bool(
        OmegaConf.select(
            config,
            "trainer.poise.estimator.eval_on_val",
            default=False,
        )
    ):
        # Preserve the shared RLOO launcher's validation sampling contract.
        return

    validation_rollouts = int(
        config.trainer.poise.validation_rollouts_per_prompt
    )
    if validation_rollouts < 1:
        raise ValueError(
            "POISE validation requires at least one rollout per prompt, got "
            f"{validation_rollouts}"
        )
    # The shared RLOO launcher keeps its validation default at one sample. POISE
    # overrides the composed value here so only the POISE entry point uses its
    # multi-sample validation contract.
    config.actor_rollout_ref.rollout.val_kwargs.n = validation_rollouts


def apply_poise_validation_file_config(config) -> None:
    """Use unique benchmarks only for estimator validation diagnostics."""
    if not bool(
        OmegaConf.select(
            config,
            "trainer.poise.estimator.eval_on_val",
            default=False,
        )
    ):
        # Preserve the shared RLOO launcher's repeated AIME/AMC files.
        return

    replacements: dict[str, str] = {}
    for repeated_name, env_name in _POISE_VALIDATION_FILE_ENV.items():
        configured_path = os.environ.get(env_name, "").strip()
        if not configured_path:
            continue
        resolved_path = Path(configured_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"POISE validation file is missing: {resolved_path}"
            )
        replacements[repeated_name] = str(resolved_path)

    if not replacements:
        return
    config.data.val_files = [
        replacements.get(Path(str(path)).name, str(path))
        for path in config.data.val_files
    ]


def run_ppo(config) -> None:
    apply_poise_validation_file_config(config)
    apply_poise_validation_rollout_config(config)
    # Custom OmegaConf resolvers are process-local. A TaskRunner serialized
    # from `python -m poise.main` does not register them in its Ray process.
    # Resolve all paths and environment defaults before crossing that boundary.
    OmegaConf.resolve(config)

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    if (
        is_cuda_available
        and OmegaConf.select(config.trainer, "profile_steps") is not None
        and len(OmegaConf.select(config.trainer, "profile_steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.trainer.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        seed = seed_poise_task_runner(config)
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print(f"TaskRunner seed: {seed}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("POISE currently supports FSDP/FSDP2 actors only")
        if config.critic.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("POISE currently supports FSDP/FSDP2 critics only")

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.workers.fsdp_workers import (
            ActorRolloutRefWorker,
            CriticWorker,
            RewardModelWorker,
        )

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node]
            * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        trainer = RayPOISETrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=ResourcePoolManager(
                resource_pool_spec=resource_pool_spec,
                mapping=mapping,
            ),
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
