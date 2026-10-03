"""Run Reasoning360's offline evaluation protocol with POISE's bundled scorers."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from .evaluation import AIME_PROMPT_SUFFIX, BENCHMARK_FILES, ensure_aime_file, json_dump, read_parquet, run_worker, summarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--model", help="Hugging Face model ID or local model directory")
    model.add_argument("--checkpoint", help="FSDP global_step_N/actor directory (merged automatically on CPU)")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="Comma-separated GPU IDs")
    parser.add_argument("--data-dir", default="data/offline_eval")
    parser.add_argument("--aime-file", help="Existing prepared AIME 2025 parquet; downloaded automatically when omitted")
    parser.add_argument("--data-files", nargs="+", help="Custom subset instead of the standard 19 benchmarks")
    parser.add_argument("--n", type=int, default=1, help="Samples per question for --data-files; standard AIME uses 32")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-prompt-tokens", type=int, default=24576)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--score-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="First N questions per benchmark, for smoke checks")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the evaluation plan without GPUs")
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config:
        return args
    if not args.model and not args.checkpoint:
        parser.error("Specify --model or --checkpoint")
    for key in ("n", "max_tokens", "max_prompt_tokens", "max_model_len", "max_num_seqs", "score_workers", "limit"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not args.data_files and args.n != 1:
        parser.error("--n applies to --data-files; the standard suite fixes N=1 / AIME N=32")
    gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if any(not gpu for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain unique, nonempty IDs")
    args.gpus = ",".join(gpu.strip() for gpu in gpus)
    args.output_dir = str(Path(args.output_dir).resolve())
    return args


def build_jobs(args):
    if args.data_files:
        pairs = [(Path(path), args.n) for path in args.data_files]
    else:
        pairs = [(Path(args.data_dir) / name, 1) for name in BENCHMARK_FILES]
        pairs.append((Path(args.aime_file or "data/aime_year_eval/rollout_prompt/math__aime_2025_30.parquet"), 32))
    if len({path.stem for path, _ in pairs}) != len(pairs):
        raise ValueError("Benchmark filenames must have unique stems")
    jobs = []
    for path, n in pairs:
        if not args.data_files and n == 32 and args.aime_file is None:
            ensure_aime_file(path)
        path = path.resolve(strict=True)
        frame = read_parquet(path, args.limit)
        if frame.empty or not {"prompt", "data_source", "reward_model"}.issubset(frame):
            raise ValueError(f"Expected nonempty GURU parquet with prompt, data_source, reward_model: {path}")
        if not args.data_files and n == 32:
            # Reject repeated AIME 2024/2026 files or a differently formatted 2025 set.
            full = read_parquet(path)
            if len(full) != 30 or set(full["year"]) != {2025} or full["problem_idx"].nunique() != 30:
                raise ValueError("Standard AIME evaluation requires 30 unique AIME 2025 questions")
            if any(row["prompt"][0]["content"] != str(row["problem"]) + AIME_PROMPT_SUFFIX for _, row in full.iterrows()):
                raise ValueError("AIME must use the training rollout prompt; omit --aime-file to prepare it automatically")
        stat = path.stat()
        jobs.append({"path": str(path), "n": n, "rows": len(frame), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return jobs


def model_identity(args):
    if args.model:
        path = Path(args.model)
        if path.is_dir():
            files = sorted([*path.glob("*.safetensors"), *path.glob("*.json"), *path.glob("*.bin")])
            return {"model": str(path.resolve()), "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]}
        return {"model": args.model}
    actor = Path(args.checkpoint).resolve(strict=True)
    if (actor / "actor").is_dir():
        actor /= "actor"
    files = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    matches = [re.fullmatch(r"model_world_size_(\d+)_rank_(\d+)\.pt", path.name) for path in files]
    sizes = {int(match[1]) for match in matches}
    ranks = {int(match[2]) for match in matches}
    if len(sizes) != 1 or ranks != set(range(next(iter(sizes), 0))) or not files:
        raise ValueError(f"Incomplete FSDP model shards: {actor}")
    if not (actor / "huggingface/config.json").is_file() or any(path.stat().st_size == 0 for path in files):
        raise ValueError(f"Missing model config or empty FSDP shard: {actor}")
    return {"checkpoint": str(actor), "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]}


def prepare_model(identity, output):
    if "model" in identity:
        return identity["model"]
    target = output / "merged_model"
    marker = target / ".complete"
    if not marker.exists():
        subprocess.run([
            sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
            "--local_dir", identity["checkpoint"], "--target_dir", str(target),
        ], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
        if not (target / "config.json").is_file() or not list(target.glob("*.safetensors")):
            raise RuntimeError("Checkpoint merger did not write a complete Hugging Face model")
        marker.touch()
    return str(target)


def launch_workers(config, gpus):
    output = Path(config["output_dir"])
    processes = []
    try:
        for slot, gpu in enumerate(gpus):
            jobs = config["jobs"][slot::len(gpus)]
            if not jobs:
                continue
            worker = f"worker{slot}"
            spec = output / "logs" / f"{worker}.json"
            json_dump(spec, {**config, "jobs": jobs, "worker": worker})
            with spec.with_suffix(".log").open("a") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "poise.eval", "--worker-config", str(spec)],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append(process)
            print(f"GPU {gpu}: {len(jobs)} benchmarks; {spec.with_suffix('.log')}", flush=True)
        failed = [process.pid for process in processes if process.wait() != 0]
        if failed:
            raise RuntimeError(f"Evaluation workers failed: {failed}; see {output / 'logs'}")
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main(argv=None):
    args = parse_args(argv)
    os.environ["CODER1_EXEC"] = "sandboxfusion"
    os.environ.setdefault("SANDBOX_FUSION_SERVERS", "127.0.0.1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.pop("RAY_ADDRESS", None)
    if args.worker_config:
        run_worker(json.loads(Path(args.worker_config).read_text()))
        return 0
    jobs = build_jobs(args)
    identity = model_identity(args)
    settings = {key: value for key, value in vars(args).items() if key not in {
        "gpus", "data_dir", "aime_file", "data_files", "dry_run", "worker_config", "model", "checkpoint", "output_dir",
    }}
    plan = {"model": identity, "settings": settings, "jobs": jobs}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "run.json"
        # Canonical JSON turns tuples into lists before comparing resumed runs.
        plan = json.loads(json.dumps(plan))
        if manifest.exists() and json.loads(manifest.read_text()) != plan:
            raise ValueError("Output directory belongs to different inputs/settings; use a new --output-dir")
        if not manifest.exists() and (list(output.glob("*.parquet")) or list(output.glob("*.metrics.json"))):
            raise ValueError("Output directory has results without a matching run.json; use a new --output-dir")
        json_dump(manifest, plan)
        if any(Path(job["path"]).name.startswith(("codegen", "simulation__cruxeval")) for job in jobs):
            import requests

            for host in os.environ["SANDBOX_FUSION_SERVERS"].split(","):
                response = requests.get(f"http://{host.strip()}:8080/v1/ping", timeout=5)
                response.raise_for_status()
                if response.json() != "pong":
                    raise RuntimeError(f"SandboxFusion health check failed: {host}")
        model = prepare_model(identity, output)
        launch_workers({**settings, "jobs": jobs, "model": model, "output_dir": str(output)}, args.gpus.split(","))
        metrics = [json.loads((output / f"{Path(job['path']).stem}.metrics.json").read_text()) for job in jobs]
        for job, metric in zip(jobs, metrics, strict=True):
            if metric["rows"] != job["rows"] or metric["samples"] != job["rows"] * job["n"]:
                raise RuntimeError(f"Incomplete evaluation: {job['path']}")
        summary = summarize(metrics)
        json_dump(output / "summary.json", summary)
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "benchmark", "score_name", "rows", "samples", "score", "pass_at_n", "clip_rate", "error_count",
            ], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metrics)
        print(json.dumps({key: value for key, value in summary.items() if key not in {"results", "domains"}}, indent=2))
        return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
