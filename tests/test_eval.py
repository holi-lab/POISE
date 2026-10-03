import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from poise.eval import build_jobs, main, model_identity, parse_args
from poise.evaluation import (
    AIME_PROMPT_SUFFIX, BENCHMARK_FILES, benchmark_metrics, expanded_source, generate, prepare_aime_rows, response_metric,
    run_worker, save_responses, score_one, summarize,
)


def source_frame():
    return pd.DataFrame([{
        "prompt": [{"role": "user", "content": "What is 6 times 7?"}],
        "data_source": "math__math", "reward_model": {"ground_truth": "42"},
    }])


def scored_frame(scores, n):
    source = pd.concat([source_frame()] * (len(scores) // n), ignore_index=True)
    frame = expanded_source(source, n)
    frame["eval_score"] = scores
    frame["eval_error"] = None
    frame["response_tokens"] = 10
    frame["response_clipped"] = False
    return frame


def test_dictionary_scorer_and_real_reasoning360_math():
    assert response_metric({"score": 0.25, "acc": 1.0}) == 0.25
    row = source_frame().iloc[0].to_dict()
    row["response"] = r"The answer is \boxed{42}."
    assert score_one(row) == (1.0, None)
    row["response"] = r"The answer is \boxed{43}."
    assert score_one(row) == (0.0, None)
    with pytest.raises(ValueError, match="Non-finite"):
        response_metric({"score": float("nan")})


def test_avg32_is_not_pass32_and_summary_weights_questions():
    # Two questions: one correct sample out of 32 on each gives avg@32=1/32, pass@32=1.
    scores = ([1.0] + [0.0] * 31) * 2
    aime = benchmark_metrics(scored_frame(scores, 32), "math__aime_2025_30", 32)
    assert aime["score"] == aime["avg_at_32"] == 1 / 32
    assert aime["pass_at_n"] == 1.0
    assert (aime["rows"], aime["samples"]) == (2, 64)
    other = benchmark_metrics(scored_frame([1, 0], 1), "codegen__example", 1)
    result = summarize([aime, other])
    assert result["rows"] == 4 and result["samples"] == 66
    assert result["weighted_score"] == result["macro_score"] == (1 / 32 + 0.5) / 2


def test_standard_suite_contains_only_aime_2025(tmp_path):
    for name in BENCHMARK_FILES:
        source_frame().to_parquet(tmp_path / name, index=False)
    aime = prepare_aime_rows([{"problem_idx": i, "problem": f"Problem {i}", "answer": i} for i in range(1, 31)])
    aime_path = tmp_path / "math__aime_2025_30.parquet"
    aime.to_parquet(aime_path, index=False)
    args = parse_args(["--model", "example", "--gpus", "0", "--data-dir", str(tmp_path), "--aime-file", str(aime_path)])
    jobs = build_jobs(args)
    assert len(jobs) == 19
    assert [job["n"] for job in jobs] == [1] * 18 + [32]
    assert [Path(job["path"]).name for job in jobs if "aime" in Path(job["path"]).name] == [aime_path.name]
    assert aime.iloc[0]["prompt"][0]["content"] == "Problem 1" + AIME_PROMPT_SUFFIX
    aime["year"] = 2024
    aime.to_parquet(aime_path, index=False)
    with pytest.raises(ValueError, match="AIME 2025"):
        build_jobs(args)


def test_aime_preparation_rejects_duplicate_questions():
    rows = [{"problem_idx": 1, "problem": "duplicate", "answer": 42}] * 30
    with pytest.raises(ValueError, match="30 unique"):
        prepare_aime_rows(rows)


def test_standard_suite_automatically_prepares_and_reuses_aime(tmp_path, monkeypatch):
    import sys

    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data/offline_eval"
    data_dir.mkdir(parents=True)
    for name in BENCHMARK_FILES:
        source_frame().to_parquet(data_dir / name, index=False)
    downloads = []

    def download(name, **kwargs):
        downloads.append((name, kwargs))
        return [{"problem_idx": i, "problem": f"Problem {i}", "answer": i} for i in range(1, 31)]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=download))
    args = parse_args(["--model", "example", "--gpus", "0", "--limit", "1"])
    jobs = build_jobs(args)
    assert len(jobs) == 19
    assert jobs[-1]["n"] == 32 and jobs[-1]["rows"] == 1
    prepared = pd.read_parquet(jobs[-1]["path"])
    assert len(prepared) == 30 and set(prepared["year"]) == {2025}
    assert prepared.iloc[0]["prompt"][0]["content"] == "Problem 1" + AIME_PROMPT_SUFFIX
    assert prepared.iloc[-1]["reward_model"]["ground_truth"] == "30"
    assert build_jobs(args) == jobs
    assert downloads == [("MathArena/aime_2025", {"split": "train", "revision": "main"})]


def test_custom_subset_and_explicit_aime_path_do_not_download(tmp_path, monkeypatch):
    import sys

    def unexpected_download(*args, **kwargs):
        pytest.fail("Custom inputs must not download the default AIME dataset")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=unexpected_download))
    for name in BENCHMARK_FILES:
        source_frame().to_parquet(tmp_path / name, index=False)
    subset = parse_args(["--model", "example", "--gpus", "0", "--data-files", str(tmp_path / BENCHMARK_FILES[0])])
    assert len(build_jobs(subset)) == 1
    explicit = parse_args(["--model", "example", "--gpus", "0", "--data-dir", str(tmp_path),
                           "--aime-file", str(tmp_path / "missing.parquet")])
    with pytest.raises(FileNotFoundError):
        build_jobs(explicit)


def test_native_chat_template_and_n_outputs(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    calls = []

    def template(chat, tokenize, add_generation_prompt):
        assert add_generation_prompt and chat[0]["role"] == "user"
        return [1, 2, 3] if tokenize else "native template"

    def sample(prompts, params, use_tqdm):
        calls.append((prompts, params))
        return [SimpleNamespace(outputs=[
            SimpleNamespace(text="first", token_ids=[1], finish_reason="stop"),
            SimpleNamespace(text="second", token_ids=[2, 3], finish_reason="length"),
        ])]

    config = dict(max_prompt_tokens=8, max_model_len=16, max_tokens=8, temperature=0.6, top_p=0.95, top_k=-1)
    frame = generate(SimpleNamespace(generate=sample), SimpleNamespace(apply_chat_template=template), source_frame(), 2, config)
    assert calls[0][0] == ["native template"] and calls[0][1]["n"] == 2
    assert frame["response"].tolist() == ["first", "second"]
    assert frame["question_index"].tolist() == [0, 0]
    assert frame["sample_index"].tolist() == [0, 1]
    assert frame["response_clipped"].tolist() == [False, True]


def test_resume_rescores_saved_responses_without_loading_model(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "math__tiny.parquet"
    source_frame().to_parquet(path, index=False)
    frame = expanded_source(source_frame(), 2)
    frame["response"] = [r"\boxed{42}", r"\boxed{43}"]
    frame["responses"] = [[value] for value in frame["response"]]
    frame["prompt_tokens"] = 10
    frame["response_tokens"] = 5
    frame["finish_reason"] = "stop"
    frame["response_clipped"] = False
    save_responses(frame, tmp_path / path.name)
    assert "reward_model" not in pd.read_parquet(tmp_path / path.name)
    config = dict(model="must-not-be-loaded", output_dir=str(tmp_path), jobs=[{"path": str(path), "n": 2}],
                  seed=1234, limit=None, score_workers=1, temperature=0.6, top_p=0.95, top_k=-1,
                  max_tokens=8, worker="test")
    run_worker(config)
    metric = json.loads((tmp_path / "math__tiny.metrics.json").read_text())
    assert metric["score"] == 0.5 and metric["error_count"] == 0
    # Completed scores resume without opening the source or importing vLLM.
    path.unlink()
    run_worker(config)


def test_output_directory_cannot_mix_settings(tmp_path):
    path = tmp_path / "math__tiny.parquet"
    source_frame().to_parquet(path, index=False)
    output = tmp_path / "out"
    output.mkdir()
    (output / "run.json").write_text(json.dumps({"settings": "different run"}))
    with pytest.raises(ValueError, match="different inputs/settings"):
        main(["--model", "example", "--gpus", "0", "--data-files", str(path), "--output-dir", str(output)])


def test_missing_fsdp_rank_is_rejected(tmp_path):
    (tmp_path / "model_world_size_2_rank_0.pt").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Incomplete FSDP"):
        model_identity(parse_args(["--checkpoint", str(tmp_path), "--gpus", "0"]))


@pytest.mark.parametrize("gpus", ["0,0", "0, 0", "0,", ""])
def test_duplicate_or_empty_gpu_selection_is_rejected(gpus):
    with pytest.raises(SystemExit):
        parse_args(["--model", "example", "--gpus", gpus])
