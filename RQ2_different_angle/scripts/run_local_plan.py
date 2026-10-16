"""Run the three local-study steps without hand-assembling commands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def invoke(arguments: list[str]) -> None:
    subprocess.run([PYTHON, *arguments], cwd=ROOT, check=True)


def run_pilot(config: str, checkpoint: str, progress_every: int, checkpoint_every: int, resume: bool) -> None:
    command = [
        "scripts/run_pilot.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--checkpoint-every",
        str(checkpoint_every),
        "--progress-every",
        str(progress_every),
    ]
    if resume and (ROOT / checkpoint).exists():
        command.extend(["--resume", checkpoint])
    invoke(command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["1", "2", "3", "4", "5"])
    parser.add_argument("--resume", action="store_true", help="resume the selected run from its checkpoint")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()

    if args.step == "1":
        invoke(["scripts/preflight_commit_mode.py"])
    elif args.step == "2":
        run_pilot(
            "configs/feasibility_20k_super_commit_500kb.json",
            "checkpoints/commit_mode_20k_super_t10000_500kb.json",
            args.progress_every,
            args.checkpoint_every,
            args.resume,
        )
        run_pilot(
            "configs/feasibility_20k_dual_commit_500kb.json",
            "checkpoints/commit_mode_20k_dual_t10000_t15000_500kb.json",
            args.progress_every,
            args.checkpoint_every,
            args.resume,
        )
    elif args.step == "3":
        run_pilot(
            "configs/local_80k_super_commit_1mb.json",
            "checkpoints/commit_mode_80k_super_t40000_1mb.json",
            args.progress_every,
            args.checkpoint_every,
            args.resume,
        )
        run_pilot(
            "configs/local_80k_dual_commit_1mb.json",
            "checkpoints/commit_mode_80k_dual_t40000_t60000_1mb.json",
            args.progress_every,
            args.checkpoint_every,
            args.resume,
        )
    elif args.step == "4":
        # Run the safer late-parity point first, so a possible underfill at
        # tParity=50k does not prevent collection of the tParity=70k result.
        run_pilot(
            "configs/local_80k_dual_t40000_t70000_1mb.json",
            "checkpoints/commit_mode_80k_dual_t40000_t70000_1mb.json",
            args.progress_every,
            args.checkpoint_every,
            args.resume,
        )
    else:
        # Run the early SuperBPE transition first.  If the later tSuper=50k
        # condition underfills its longer Csub phase, this valid result remains.
        run_pilot(
            "configs/local_80k_dual_t30000_t60000_1mb.json",
            "checkpoints/commit_mode_80k_dual_t30000_t60000_1mb.json",
            args.progress_every,
            args.checkpoint_every,
            args.resume,
        )
        run_pilot(
            "configs/local_80k_dual_t50000_t60000_1mb.json",
            "checkpoints/commit_mode_80k_dual_t50000_t60000_1mb.json",
            args.progress_every,
            args.checkpoint_every,
            args.resume,
        )
        run_pilot(
            "configs/local_80k_dual_t40000_t50000_1mb.json",
            "checkpoints/commit_mode_80k_dual_t40000_t50000_1mb.json",
            args.progress_every,
            args.checkpoint_every,
            args.resume,
        )


if __name__ == "__main__":
    main()
