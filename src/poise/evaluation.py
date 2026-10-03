"""Generation and scoring adapted from Reasoning360's offline_eval workers."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARK_FILES = (
    "codegen__humaneval_164.parquet", "codegen__livecodebench_279.parquet", "codegen__mbpp_500.parquet",
    "logic__arcagi1_400.parquet", "logic__zebra_puzzle_dataset_200.parquet", "math__math_500.parquet",
    "ood__ifeval_541.parquet", "ood__livebench_data_analysis_150.parquet",
    "ood__livebench_language_140.parquet", "ood__livebench_reasoning_150.parquet",
    "simulation__codeio_200.parquet", "simulation__cruxeval-i_800.parquet", "simulation__cruxeval-o_800.parquet",
    "stem__gpqa_diamond_198.parquet", "stem__supergpqa_1k.parquet",
    "table__finqa_1.1k.parquet", "table__hitab_1k.parquet", "table__multihier_336.parquet",
)
GENERATED_COLUMNS = (
    "question_index", "sample_index", "response", "responses", "prompt_tokens", "response_tokens",
    "finish_reason", "response_clipped", "eval_score", "eval_error",
)
INPUT_COLUMNS = ("year", "problem_idx", "problem", "data_source", "prompt", "ability", "extra_info")
AIME_PROMPT_SUFFIX = " Please output the final answer within \\boxed{}."


def prepare_aime_rows(source):
    if len(source) != 30 or sorted(int(row["problem_idx"]) for row in source) != list(range(1, 31)):
        raise ValueError("AIME 2025 must contain exactly 30 unique problem_idx values, 1 through 30")
    rows = []
    for index, row in enumerate(sorted(source, key=lambda row: int(row["problem_idx"]))):
        if "year" in row and int(row["year"]) != 2025:
            raise ValueError("Only AIME 2025 is part of this evaluation suite")
        problem = str(row["problem"]).strip()
        answer = str(row["reward_model"]["ground_truth"] if "reward_model" in row else row["answer"])
        problem_idx = int(row["problem_idx"])
        rows.append({
            "year": 2025, "problem_idx": problem_idx, "problem": problem, "data_source": "math__aime_2025",
            "prompt": [{"role": "user", "content": problem + AIME_PROMPT_SUFFIX}],
            "ability": "math", "apply_chat_template": True,
            "reward_model": {"ground_truth": answer, "style": "rule"},
            "extra_info": {"index": index, "problem_idx": problem_idx, "original_question": problem,
                           "reward_metric": "default", "split": "test", "year": 2025},
        })
    return pd.DataFrame(rows)


def ensure_aime_file(path):
    """Prepare the default AIME 2025 dataset once, then reuse the local parquet."""
    if path.is_file():
        return
    from datasets import load_dataset

    frame = prepare_aime_rows(list(load_dataset("MathArena/aime_2025", split="train", revision="main")))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".parquet.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_parquet(path, limit=None):
    # The published LiveCodeBench nested schema needs Polars, as in Reasoning360.
    if "livecodebench" in Path(path).name:
        import polars as pl

        frame = pl.scan_parquet(path)
        if limit is not None:
            frame = frame.head(limit)
        return pd.DataFrame(frame.collect().to_dicts())
    frame = pd.read_parquet(path)
    return frame.head(limit).copy() if limit is not None else frame


def save_responses(frame, path):
    columns = [key for key in (*INPUT_COLUMNS, *GENERATED_COLUMNS) if key in frame]
    temporary = path.with_suffix(".parquet.tmp")
    # Reattach ground truth from the source when resuming; don't duplicate large code test payloads.
    frame[columns].to_parquet(temporary, index=False)
    os.replace(temporary, path)


def response_metric(result):
    value = float(result["score"] if isinstance(result, dict) else result)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite evaluation score: {value}")
    return value


def score_one(row):
    from verl.utils.reward_score import default_compute_score

    try:
        reward = row["reward_model"]
        extra = row.get("extra_info")
        if isinstance(reward, np.ndarray) and reward.ndim == 0:
            reward = reward.item()
        if isinstance(extra, np.ndarray) and extra.ndim == 0:
            extra = extra.item()
        result = default_compute_score(str(row["data_source"]), str(row["response"]), reward["ground_truth"], extra)
        return response_metric(result), None
    except Exception as error:
        return 0.0, f"{type(error).__name__}: {error}"


def score_frame(frame, workers):
    columns = [key for key in ("data_source", "response", "reward_model", "extra_info") if key in frame]
    records = frame[columns].to_dict("records")
    parallel = str(frame.iloc[0]["data_source"]).startswith(("codegen", "simulation__cruxeval"))
    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(score_one, records))
    else:
        results = [score_one(row) for row in records]
    scores, errors = zip(*results, strict=True)
    return list(scores), list(errors)


def expanded_source(source, n):
    frame = source.reset_index(drop=True).loc[np.repeat(np.arange(len(source)), n)].reset_index(drop=True)
    frame["question_index"] = np.repeat(np.arange(len(source)), n)
    frame["sample_index"] = np.tile(np.arange(n), len(source))
    return frame


def generate(llm, tokenizer, source, n, config):
    from vllm import SamplingParams

    messages = [value.tolist() if isinstance(value, np.ndarray) else value for value in source["prompt"]]
    tokens = [len(tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)) for chat in messages]
    if max(tokens) > config["max_prompt_tokens"] or max(tokens) + config["max_tokens"] > config["max_model_len"]:
        raise ValueError(f"Prompt requires {max(tokens)} tokens; exceeds the configured prompt/context budget")
    texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in messages]
    sampling = SamplingParams(n=n, **{key: config[key] for key in ("temperature", "top_p", "top_k", "max_tokens")})
    requests = llm.generate(texts, sampling, use_tqdm=True)
    if len(requests) != len(source) or any(len(request.outputs) != n for request in requests):
        raise RuntimeError("Incomplete generation: expected N outputs for every question")
    outputs = [output for request in requests for output in request.outputs]
    frame = expanded_source(source, n)
    frame["response"] = [output.text for output in outputs]
    frame["responses"] = [[output.text] for output in outputs]
    frame["prompt_tokens"] = np.repeat(tokens, n)
    frame["response_tokens"] = [len(output.token_ids) for output in outputs]
    frame["finish_reason"] = [output.finish_reason for output in outputs]
    frame["response_clipped"] = [output.finish_reason == "length" for output in outputs]
    return frame


def benchmark_metrics(frame, benchmark, n):
    questions = frame.groupby("question_index", sort=True)["eval_score"].agg(["mean", "sum", "count"])
    if not (questions["count"] == n).all():
        raise ValueError("Every question must have exactly N scored responses")
    score = float(questions["mean"].mean())
    tokens = frame["response_tokens"].astype(int)
    errors = [error for error in frame["eval_error"] if error is not None]
    return {
        "benchmark": benchmark, "data_source": str(frame.iloc[0]["data_source"]),
        "rows": len(questions), "samples": len(frame), "samples_per_question": n, "n": n,
        "score": score, "score_name": "accuracy" if n == 1 else f"avg@{n}",
        "correct": float(questions["mean"].sum()), "correct_samples": float(frame["eval_score"].sum()),
        "avg_at_n": score, "avg_at_32": score if n == 32 else None,
        "pass_at_n": float((questions["sum"] > 0).mean()),
        "response_tokens_mean": float(tokens.mean()), "response_tokens_p95": float(tokens.quantile(0.95)),
        "response_tokens_max": int(tokens.max()), "clip_count": int(frame["response_clipped"].sum()),
        "clip_rate": float(frame["response_clipped"].mean()), "error_count": len(errors), "error_examples": errors[:5],
        "per_question": [
            {"question_index": int(index), "correct": float(row["sum"]), "samples": int(row["count"]),
             "avg_at_n": float(row["mean"])} for index, row in questions.iterrows()
        ],
    }


def summarize(metrics):
    def weighted(items):
        return sum(item["score"] * item["rows"] for item in items) / sum(item["rows"] for item in items)

    domains = {}
    for item in metrics:
        domains.setdefault(item["benchmark"].split("__", 1)[0], []).append(item)
    return {
        "benchmarks": len(metrics), "rows": sum(item["rows"] for item in metrics),
        "samples": sum(item["samples"] for item in metrics),
        "macro_score": float(np.mean([item["score"] for item in metrics])), "weighted_score": weighted(metrics),
        "avg_at_32": {item["benchmark"]: item["avg_at_32"] for item in metrics if item["avg_at_32"] is not None},
        "clip_rate": sum(item["clip_count"] for item in metrics) / sum(item["samples"] for item in metrics),
        "error_count": sum(item["error_count"] for item in metrics),
        "domains": {
            domain: {"benchmarks": len(items), "rows": sum(item["rows"] for item in items),
                     "samples": sum(item["samples"] for item in items), "weighted_score": weighted(items)}
            for domain, items in sorted(domains.items())
        },
        "results": metrics,
    }


def shutdown_llm(llm):
    engine = llm.llm_engine
    for owner in (engine, getattr(engine, "engine_core", None)):
        shutdown = getattr(owner, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
    raise RuntimeError("Unable to locate vLLM shutdown method")


def run_worker(config):
    import random

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    llm = None
    tokenizer = None
    output_dir = Path(config["output_dir"])
    try:
        for job in config["jobs"]:
            path = Path(job["path"])
            metric_path = output_dir / f"{path.stem}.metrics.json"
            if metric_path.exists() and json.loads(metric_path.read_text())["error_count"] == 0:
                print(f"Already scored: {path.name}", flush=True)
                continue
            source = read_parquet(path, config["limit"])
            output_path = output_dir / path.name
            started = time.monotonic()
            if output_path.exists():
                frame = expanded_source(source, job["n"])
                cached = read_parquet(output_path)
                required = set(GENERATED_COLUMNS) - {"eval_score", "eval_error"}
                if len(cached) != len(frame) or not required.issubset(cached.columns):
                    raise ValueError(f"Incomplete cached responses: {output_path}")
                if not np.array_equal(cached[["question_index", "sample_index"]], frame[["question_index", "sample_index"]]):
                    raise ValueError(f"Cached response order mismatch: {output_path}")
                for key in GENERATED_COLUMNS:
                    if key in cached:
                        frame[key] = cached[key].tolist()
            else:
                if llm is None:
                    from transformers import AutoTokenizer
                    from vllm import LLM

                    tokenizer = AutoTokenizer.from_pretrained(config["model"])
                    tokenizer.model_max_length = config["max_model_len"]
                    llm = LLM(
                        model=config["model"], dtype="bfloat16", tensor_parallel_size=1,
                        gpu_memory_utilization=config["gpu_memory_utilization"], enforce_eager=config["enforce_eager"],
                        max_model_len=config["max_model_len"], max_num_batched_tokens=config["max_model_len"],
                        max_num_seqs=config["max_num_seqs"], enable_chunked_prefill=True, seed=config["seed"],
                    )
                frame = generate(llm, tokenizer, source, job["n"], config)
                save_responses(frame, output_path)
            generation_seconds = time.monotonic() - started
            started = time.monotonic()
            frame["eval_score"], frame["eval_error"] = score_frame(frame, config["score_workers"])
            scoring_seconds = time.monotonic() - started
            save_responses(frame, output_path)
            metric = benchmark_metrics(frame, path.stem, job["n"])
            metric.update({key: config[key] for key in ("seed", "temperature", "top_p", "top_k", "max_tokens", "worker")})
            metric.update(generation_seconds=generation_seconds, scoring_seconds=scoring_seconds)
            json_dump(metric_path, metric)
            print("METRIC " + json.dumps(metric), flush=True)
    finally:
        if llm is not None:
            shutdown_llm(llm)
