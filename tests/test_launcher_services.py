"""Exercise launcher process ownership and exit handling without GPUs or services."""

import json
import os
import subprocess
import sys

import pytest

from launcher_support import ROOT, launcher_environment


@pytest.fixture
def launch_environment(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def executable(name, body):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)
        return path

    python_bin = executable("python", '''import json, os, sys
from pathlib import Path
root = Path(os.environ["STUB_DIR"])
if sys.argv[1:3] == ["-m", "poise.main"]:
    keys = ["STEM_LLM_JUDGE_URL", "STEM_VERIFIER_LIFECYCLE_CONTROL", "STEM_VERIFIER_LIFECYCLE_URL"]
    (root / "training.json").write_text(json.dumps({"argv": sys.argv[1:], "env": {k: os.getenv(k) for k in keys}}))
    print("training stub", flush=True)
    sys.exit(int(os.environ["STUB_EXIT_CODE"]))
if sys.argv[1:3] == ["-m", "verl.utils.reward_score.stem_llm_judge.lifecycle"]:
    (root / "lifecycle.json").write_text(json.dumps(sys.argv[3:]))
else:
    raise SystemExit(f"Unexpected Python invocation: {sys.argv}")
''')
    vllm_bin = executable("vllm", '''import os, signal
from pathlib import Path
root = Path(os.environ["STUB_DIR"])
def stop(*args):
    (root / "verifier.stopped").touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root / "verifier.pid").write_text(str(os.getpid()))
while True:
    signal.pause()
''')
    executable("curl", '''import os, sys, time
from pathlib import Path
url = sys.argv[-1]
if url.endswith("/v1/ping"):
    print('"pong"')
elif url.endswith("/v1/models"):
    if os.environ["STUB_SERVICE"] == "local":
        marker = Path(os.environ["STUB_DIR"]) / "verifier.pid"
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.01)
        else:
            raise SystemExit("Verifier did not start")
else:
    raise SystemExit(f"Unexpected request: {url}")
''')
    recipe = json.loads((ROOT / "tests/fixtures/launcher_contracts.json").read_text())["qwen3_4b"]
    for split, key in [("train", "data.train_files"), ("val", "data.val_files")]:
        directory = tmp_path / split
        directory.mkdir()
        for name in recipe["arguments"][key]:
            (directory / name).touch()
    environment = launcher_environment(tmp_path, python_bin)
    environment.pop("STEM_LLM_JUDGE_URL", None)
    environment.update(
        CONFIG_ONLY="0", PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        TRAIN_DATA_DIR=str(tmp_path / "train"), VAL_DATA_DIR=str(tmp_path / "val"),
        VLLM_BIN=str(vllm_bin), STUB_DIR=str(tmp_path), STUB_SERVICE="local", STUB_EXIT_CODE="0",
        AUTO_START_STEM_VERIFIER="1", STEM_VERIFIER_PORT="18080", STEM_VERIFIER_STARTUP_POLLS="1",
        STEM_VERIFIER_SLEEP_DURING_NON_REWARD="1", STEM_VERIFIER_LIFECYCLE_URL="http://stale-service",
        LOG_FILE=str(tmp_path / "logs/train.log"), STEM_VERIFIER_LOG=str(tmp_path / "logs/verifier.log"),
    )
    return environment


@pytest.mark.parametrize("service", ["local", "external"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_training_preserves_exit_status_and_verifier_ownership(tmp_path, launch_environment, service, exit_code):
    environment = launch_environment
    environment.update(STUB_SERVICE=service, STUB_EXIT_CODE=str(exit_code))
    if service == "external":
        environment["STEM_LLM_JUDGE_URL"] = "http://external-verifier"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train/poise_qwen3_4b.sh"), "trainer.save_freq=1"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    training = json.loads((tmp_path / "training.json").read_text())
    assert training["argv"][-1] == "trainer.save_freq=1"
    assert "--cfg" not in training["argv"]
    assert (tmp_path / "logs/train.log").read_text().strip() == "training stub"
    if service == "local":
        assert training["env"]["STEM_VERIFIER_LIFECYCLE_CONTROL"] == "1"
        assert json.loads((tmp_path / "lifecycle.json").read_text())[0] == "sleep"
        assert (tmp_path / "verifier.stopped").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "verifier.pid").read_text()), 0)
    else:
        assert training["env"] == {
            "STEM_LLM_JUDGE_URL": "http://external-verifier",
            "STEM_VERIFIER_LIFECYCLE_CONTROL": "0",
            "STEM_VERIFIER_LIFECYCLE_URL": None,
        }
        assert not (tmp_path / "lifecycle.json").exists()
        assert not (tmp_path / "verifier.pid").exists()


def test_missing_data_stops_before_starting_services(tmp_path, launch_environment):
    next((tmp_path / "train").iterdir()).unlink()
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train/poise_qwen3_4b.sh")],
        cwd=tmp_path, env=launch_environment, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 2
    assert "Missing dataset:" in result.stderr
    assert not (tmp_path / "training.json").exists()
    assert not (tmp_path / "verifier.pid").exists()
    assert not (tmp_path / "output").exists()
