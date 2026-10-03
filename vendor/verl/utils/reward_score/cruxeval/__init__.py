import json
import re
from typing import Any

from verl.utils.reward_score.coder1 import code_exec


def compute_score(model_output: str, ground_truth: str, extra_info: Any = None) -> dict[str, int]:
    model_output = str(model_output)
    if "</think>" in model_output:
        model_output = model_output.rsplit("</think>", maxsplit=1)[-1]
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", model_output, re.DOTALL)
    if code_blocks:
        model_output = code_blocks[-1].strip()

    full_code = json.loads(ground_truth)["functional"] + "\n" + model_output
    # print(f">>> {full_code}")
    success, _ = code_exec(full_code, timeout=3)
    is_correct = 1 if success else 0
    # print(f">>> {is_correct}")
    return {"score": is_correct, "acc": is_correct}
