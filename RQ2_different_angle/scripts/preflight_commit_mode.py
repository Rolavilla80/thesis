"""Verify the commit-compatible mode and print the planned curricula."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from different_angle.pretokenize import split_text
from different_angle.schedule import Phase, three_phase_schedule

CONFIGS = [
    "feasibility_20k_super_commit_500kb.json",
    "feasibility_20k_dual_commit_500kb.json",
    "local_80k_super_commit_1mb.json",
    "local_80k_dual_commit_1mb.json",
    "local_80k_dual_t40000_t50000_1mb.json",
    "local_80k_dual_t40000_t70000_1mb.json",
    "local_80k_dual_t30000_t60000_1mb.json",
    "local_80k_dual_t50000_t60000_1mb.json",
]


def schedule(config: dict) -> list[Phase]:
    if "phases" in config:
        return [Phase(**phase) for phase in config["phases"]]
    transitions = config["transitions"]
    return three_phase_schedule(
        config["vocab_size"], transitions["super_vocab"], transitions["parity_vocab"]
    )


def main() -> None:
    probe = "Hello world!"
    expected_sub = (b" Hello", b" world", b" !")
    expected_super = (b" Hello world!",)
    assert split_text(probe, "sub", "commit_bytelevel") == expected_sub
    assert split_text(probe, "super", "commit_bytelevel") == expected_super
    print("commit_bytelevel pretokenizer check: passed")

    for filename in CONFIGS:
        config = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))
        if config["pretokenizer"] != "commit_bytelevel":
            raise ValueError(f"{filename} does not use commit_bytelevel")
        if config["parity_reference"] != "phase":
            raise ValueError(f"{filename} does not use phase-rebased parity")
        train_dir = ROOT / config["data_contract"]["train_dir"]
        missing = [language for language in config["languages"] if not (train_dir / f"{language}.txt").exists()]
        if missing:
            raise FileNotFoundError(f"{filename} is missing training files: {', '.join(missing)}")
        phases = schedule(config)
        print(f"{filename}: " + " -> ".join(f"{phase.name}@{phase.n_merges}" for phase in phases))


if __name__ == "__main__":
    main()
