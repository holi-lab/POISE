"""Download GURU parquet splits for training and evaluation."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--splits", nargs="+", choices=["train", "online_eval", "offline_eval"],
                        default=["train", "online_eval"])
    args = parser.parse_args()
    snapshot_download(
        "LLM360/guru-RL-92k",
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=[f"{split}/*.parquet" for split in args.splits],
    )


if __name__ == "__main__":
    main()
