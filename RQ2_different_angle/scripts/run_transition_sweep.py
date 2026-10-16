"""Run a named transition sweep through the same single-experiment runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=ROOT / "configs" / "transition_sweep_2k_balanced.json"
    )
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    base_config = ROOT / spec["base_config"]
    for condition in spec["conditions"]:
        print(f"\n=== sweep condition: {condition['name']} ===", flush=True)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_pilot.py"),
            "--config",
            str(base_config),
            "--name",
            condition["name"],
            "--super-vocab",
            str(condition["super_vocab"]),
            "--parity-vocab",
            str(condition["parity_vocab"]),
            "--progress-every",
            str(args.progress_every),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
