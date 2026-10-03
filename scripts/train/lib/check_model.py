"""Check model architecture, runtime versions, context length, and chat template."""

import sys
from importlib.metadata import version

from transformers import AutoConfig, AutoTokenizer
from vllm import __version__ as vllm_version
from vllm.model_executor.models.registry import ModelRegistry


def main():
    model_path = sys.argv[1]
    requested_context = int(sys.argv[2])
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    architectures = list(config.architectures or [])
    ray_version = version("ray")
    unsupported = [architecture for architecture in architectures if architecture not in ModelRegistry.get_supported_archs()]
    if unsupported:
        raise SystemExit(
            f"vLLM {vllm_version} does not support {unsupported} from {model_path}. "
            "Install requirements/train.txt in the environment used by PYTHON_BIN."
        )
    if vllm_version != "0.11.0" or ray_version != "2.56.1":
        raise SystemExit(
            f"Qwen3 and OLMo3 require the pinned POISE pair vLLM 0.11.0 / Ray 2.56.1; "
            f"found vLLM {vllm_version} / Ray {ray_version}. "
            "Install requirements/train.txt in the environment used by PYTHON_BIN."
        )

    model_context = getattr(config, "max_position_embeddings", None)
    if model_context is not None and requested_context > model_context:
        raise SystemExit(
            f"Requested context {requested_context} exceeds {model_path}'s max_position_embeddings={model_context}."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise SystemExit(f"Tokenizer for {model_path} has no chat template.")
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "preflight"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    print(
        f"Model preflight passed: model={model_path} architecture={architectures} "
        f"context={model_context} vllm={vllm_version} ray={ray_version}"
    )


if __name__ == "__main__":
    main()
