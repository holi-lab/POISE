"""Exercise the complete 16-step bootstrap transition with synthetic CPU workers.

The POISE loop, feature builder, Ridge fits, FIFO state, and estimator checkpoint
I/O are real. Generation and the distributed actor are replaced with tiny workers.
"""

from types import SimpleNamespace

from hydra import compose, initialize_config_module
import numpy as np
import pytest
import torch

from poise.domain_estimator_bank import DOMAIN_NAMES, DomainEstimatorBank
from poise.main import RayPOISETrainer
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [99] if text == "</think>" else [3, 4]

    def decode(self, ids, **kwargs):
        return "answer"

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row) for row in rows]

    def __call__(self, text, **kwargs):
        return {"input_ids": [3, 4], "offset_mapping": [(0, 3), (3, 6)]}


class TinyWorkers:
    def __init__(self, trainer):
        self.trainer = trainer
        self.updates = []
        self.generation_calls = 0
        self.parameter = torch.nn.Parameter(torch.tensor(0.1))
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.01)

    def generate_sequences(self, batch):
        self.generation_calls += 1
        prompts = batch.batch["input_ids"]
        responses = torch.tensor([[3, 4, 2]]).repeat(len(prompts), 1)
        ids = torch.cat([prompts, responses], dim=1)
        return DataProto.from_dict(
            tensors={"prompts": prompts, "responses": responses, "input_ids": ids,
                     "attention_mask": torch.ones_like(ids),
                     "position_ids": torch.arange(ids.shape[1]).repeat(len(ids), 1)},
            meta_info={"timing": {}},
        )

    def compute_log_prob(self, batch):
        size = len(batch)
        hidden = torch.arange(size * 4, dtype=torch.float32).reshape(size, 4) / 10
        return DataProto.from_dict(tensors={
            "old_log_probs": torch.full((size, 3), -1.0),
            "entropys": torch.full((size, 3), 0.5),
            "estimator_prompt_hidden": hidden,
            "estimator_response_hidden": hidden + 0.25,
        })

    def update_actor(self, batch):
        step = self.trainer.global_steps
        advantages = batch.batch["advantages"].clone()
        self.updates.append((step, self.trainer._poise_phase, advantages))
        assert torch.isfinite(advantages).all()
        assert "estimator_prompt_hidden" not in batch.batch
        self.optimizer.zero_grad()
        loss = -(self.parameter * advantages).mean()
        loss.backward()
        self.optimizer.step()
        return DataProto(meta_info={"metrics": {"actor/test_loss": [float(loss.detach())]}})


@pytest.mark.parametrize("reward_pattern", ["mixed", "all_zero", "all_one"])
def test_full_loop_switches_after_sixteen_updates_and_restores_estimators(tmp_path, monkeypatch, reward_pattern):
    with initialize_config_module(config_module="poise.config", version_base=None):
        config = compose(config_name="poise_fsdp", overrides=[
            "data.train_batch_size=3", "data.gen_batch_size=3",
            "trainer.total_epochs=1", "trainer.total_training_steps=17",
            "trainer.balance_batch=false", "trainer.val_before_train=false",
            "trainer.save_freq=-1", "trainer.test_freq=-1", "trainer.logger=[]",
            f"trainer.default_local_dir={tmp_path}", "trainer.poise.estimator.save_freq=17",
        ])
    trainer = object.__new__(RayPOISETrainer)
    trainer.config = config
    trainer.tokenizer = TinyTokenizer()
    trainer.total_training_steps = 17
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.use_reference_policy = trainer.use_critic = trainer.use_rm = False
    trainer.val_reward_fn = None
    trainer._load_checkpoint = lambda: None
    base_checkpoints = []
    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", lambda self: base_checkpoints.append(self.global_steps))

    def reward(batch, return_dict=False):
        values = torch.zeros((len(batch), 3))
        if reward_pattern == "mixed":
            values[:, -1] = torch.arange(len(batch)) % 2
        elif reward_pattern == "all_one":
            values[:, -1] = 1.0
        return {"reward_tensor": values, "reward_extra_info": {}} if return_dict else values

    trainer.reward_fn = reward
    trainer.actor_rollout_wg = TinyWorkers(trainer)
    trainer.train_dataloader = []
    for step in range(17):
        ids = torch.tensor([[10 + step, 20], [10 + step, 21], [10 + step, 22]])
        trainer.train_dataloader.append({
            "input_ids": ids, "attention_mask": torch.ones_like(ids),
            "position_ids": torch.tensor([[0, 1]]).repeat(3, 1),
            "raw_prompt_ids": np.array([[10 + step, 20], [10 + step, 21], [10 + step, 22]], dtype=object),
            "data_source": np.array(["math_test", "codegen_test", "stem_test"], dtype=object),
        })

    trainer.fit()
    updates = trainer.actor_rollout_wg.updates
    assert len(updates) == 17
    assert trainer.actor_rollout_wg.generation_calls == 17
    assert all(advantages.shape == (6, 3) for _, _, advantages in updates)
    assert [phase for _, phase, _ in updates[:16]] == ["bootstrap"] * 16
    assert updates[16][0:2] == (17, "poise")
    expected = torch.tensor([-1., 1., -1., 1., -1., 1.]) if reward_pattern == "mixed" else torch.zeros(6)
    for _, _, advantages in updates[:16]:
        torch.testing.assert_close(advantages[:, 0], expected)
    assert base_checkpoints == [16]
    assert (tmp_path / "global_step_16/poise_phase_state.pt").is_file()
    assert (tmp_path / "global_step_16/poise_domain_estimator_state.pt").is_file()
    assert not (tmp_path / "prompt_reward_logs/16.json").exists()
    assert (tmp_path / "prompt_reward_logs/17.json").is_file()

    restored = DomainEstimatorBank.from_checkpoint(
        tmp_path / "adaptive_estimator_updates/global_step_17",
        **trainer._estimator_runtime_kwargs(),
    )
    for domain in DOMAIN_NAMES:
        assert len(restored.states[domain].flattened_rows()) == 34
        assert restored.states[domain].retrain_count == 2
        assert restored.states[domain].observed_steps == 17
        row = restored.states[domain].flattened_rows()[-1]
        features = {key: row[key] for key in ("prompt_hidden", "response_hidden", "response_features")}
        assert restored.estimator_for(domain).predict_value(**features) == trainer._poise_bank.estimator_for(domain).predict_value(**features)
