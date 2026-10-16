#!/usr/bin/env python3
"""Benchmark UTF-8 merge-check baseline strategies for parity-aware BPE."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_once(
    root: Path,
    learner_script: Path,
    strategy: str,
    symbols: int,
    variant: str,
    train_files: list[Path],
    dev_files: list[Path],
    output_dir: Path,
    omit_strategy_flag: bool,
) -> float:
    script_stem = learner_script.stem.replace("parity_aware_learn_bpe", "pabpe")
    output_path = output_dir / f"merges.{script_stem}.{variant}.{strategy}.txt"
    cmd = [
        sys.executable,
        str(learner_script),
        "--symbols",
        str(symbols),
        "--variant",
        variant,
        "--input",
        *(str(path) for path in train_files),
        "--dev",
        *(str(path) for path in dev_files),
        "--output",
        str(output_path),
    ]
    if not omit_strategy_flag:
        cmd[6:6] = ["--utf8-merge-check-strategy", strategy]

    start = time.perf_counter()
    subprocess.run(cmd, cwd=root, check=True)
    end = time.perf_counter()
    return end - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark UTF-8 merge-check baseline strategies.")
    parser.add_argument(
        "--symbols",
        type=int,
        default=100,
        help="Number of BPE merges to learn per run. Default: %(default)s",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "window"],
        help="Parity-aware BPE variant to benchmark. Default: %(default)s",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated runs per strategy. Default: %(default)s",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["none", "max", "rescan", "block45"],
        choices=["none", "max", "rescan", "rule1",
                 "block3", "block4", "block5", "block34", "block35", "block45", "block345",
                 "char_first"],
        help="Strategies to benchmark. Default: %(default)s",
    )
    parser.add_argument(
        "--learner-script",
        type=Path,
        default=Path("parity_aware_bpe") / "parity_aware_learn_bpe.py",
        help="Learner script to execute. Default: %(default)s",
    )
    parser.add_argument(
        "--omit-strategy-flag",
        action="store_true",
        help="Do not pass --utf8-merge-check-strategy to the learner. Use this for fixed-strategy learner copies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for merge outputs. Default: benchmark_outputs/",
    )
    parser.add_argument(
        "--train",
        nargs="*",
        type=Path,
        default=None,
        help="Optional training files. Default: data_tiny/train.en data_tiny/train.de",
    )
    parser.add_argument(
        "--dev",
        nargs="*",
        type=Path,
        default=None,
        help="Optional dev files. Default: data_tiny/dev.en data_tiny/dev.de",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    train_files = args.train or [
        root / "data_tiny" / "train.en",
        root / "data_tiny" / "train.de",
    ]
    dev_files = args.dev or [
        root / "data_tiny" / "dev.en",
        root / "data_tiny" / "dev.de",
    ]
    output_dir = args.output_dir or (root / "benchmark_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    learner_script = args.learner_script
    if not learner_script.is_absolute():
        learner_script = root / learner_script

    if not learner_script.exists():
        raise FileNotFoundError(f"Missing learner script: {learner_script}")

    for path in [*train_files, *dev_files]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print(
        f"Benchmarking strategies {args.strategies} with symbols={args.symbols}, "
        f"variant={args.variant}, repeats={args.repeats}, learner={learner_script}",
        flush=True,
    )

    for strategy in args.strategies:
        durations = []
        for repeat in range(args.repeats):
            duration = run_once(
                root=root,
                learner_script=learner_script,
                strategy=strategy,
                symbols=args.symbols,
                variant=args.variant,
                train_files=train_files,
                dev_files=dev_files,
                output_dir=output_dir,
                omit_strategy_flag=args.omit_strategy_flag,
            )
            durations.append(duration)
            print(
                f"{strategy:>7} run {repeat + 1}/{args.repeats}: {duration:.4f}s",
                flush=True,
            )

        mean_duration = statistics.fmean(durations)
        print(
            f"{strategy:>7} summary: min={min(durations):.4f}s "
            f"mean={mean_duration:.4f}s max={max(durations):.4f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
