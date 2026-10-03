# POISE

Official implementation of **[Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States](https://arxiv.org/abs/2605.07579)**.

**Accepted at NeurIPS 2026** · [Project page](https://holi-lab.github.io/POISE/)

## Motivation

Estimating a baseline for policy gradient updates often requires a separate critic or many rollouts per prompt, adding memory and sampling costs. **POISE predicts the baseline from the actor's own internal states** with a lightweight probe, without training a separate critic or averaging rewards over a large rollout group.

## Method

**POISE (Policy Optimization with Internal State Value Estimation)** learns a lightweight value probe from the actor's hidden states and token entropy. For two independent rollouts of a prompt, each response uses the other response's predicted value as its baseline:

$$
A_1 = r_1 - g_\phi(z_2), \qquad A_2 = r_2 - g_\phi(z_1).
$$

Here, $z_i$ contains rollout $i$'s internal-state features, and $r_i$ is its reward. The probe learns from the paired targets $(z_1, r_2)$ and $(z_2, r_1)$.

The implementation uses **PCA + ridge regression**, with separate recent-trajectory buffers for math, code, and other domains. Advantages are fixed before the online probe update, then used in a clipped PPO policy update. Hidden states are collected during the actor's log-probability forward pass.

Code: [cross-rollout advantages](src/poise/core.py) · [value probe](src/poise/estimator.py) · [domain buffers](src/poise/domain_estimator_bank.py) · [training loop](src/poise/trainer.py).

## Setup

Linux, **Python 3.12**, and a CUDA toolkit compatible with **PyTorch 2.8.0**. The default launch configuration targets **2 × H100 80 GB** GPUs and uses **vLLM 0.11.0**.

```bash
git clone https://github.com/holi-lab/POISE.git poise
cd poise
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -r requirements/train.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e '.[train]'
```

## Training

Download [GURU](https://huggingface.co/datasets/LLM360/guru-RL-92k):

```bash
python scripts/data/download_guru.py --output-dir data
```

Start [SandboxFusion](https://github.com/bytedance/SandboxFusion) at `http://127.0.0.1:8080` for code rewards. The launcher starts the STEM verifier automatically; set `STEM_LLM_JUDGE_URL` to use an existing service.

Choose a model:

```bash
# Qwen3-4B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/poise_qwen3_4b.sh

# OLMo-3-7B-Instruct-DPO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/poise_olmo3_7b_instruct_dpo.sh
```

See [configuration](docs/configuration.md) for overrides, logging, and resume. Model-specific probe settings are in [`src/poise/config/`](src/poise/config/).

## Evaluation

We evaluate on **19 benchmarks** covering the paper's six domains: **mathematics, code generation, STEM, logic, simulation, and tabular reasoning**.

Keep SandboxFusion running for code scoring.

```bash
python scripts/data/download_guru.py --splits offline_eval
bash scripts/eval/poise.sh --checkpoint path/to/global_step_N/actor \
  --gpus 0,1,2,3,4,5,6,7 --output-dir outputs/eval
```

For a base model, replace `--checkpoint ...` with `--model Qwen/Qwen3-4B` or `--model allenai/Olmo-3-7B-Instruct-DPO`.
Results are saved as `summary.json` and `summary.csv` in the output directory.

## Citation

```bibtex
@misc{choi2026poise,
  title={Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States},
  author={Yunho Choi and Jongwon Lim and Woojin Ahn and Minjae Oh and Jeonghoon Shim and Yohan Jo},
  year={2026},
  eprint={2605.07579},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.07579}
}
```

## Acknowledgments

Built on [veRL](https://github.com/volcengine/verl) and [Reasoning360](https://github.com/LLM360/Reasoning360). Licensed under [Apache 2.0](LICENSE); see [NOTICE](NOTICE) for upstream attribution.
