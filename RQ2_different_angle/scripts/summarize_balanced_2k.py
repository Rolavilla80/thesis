"""Create one compact, reproducible table for the completed balanced 2k pilot."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "balanced_2k_pilot_summary"

RUNS = [
    ("Classical BPE", "balanced_2k_classical"),
    ("PA-BPE", "balanced_2k_parity"),
    ("SuperBPE (tS=1000)", "balanced_2k_super_t1000"),
    ("Dual (tS=1000, tP=1000)", "balanced_2k_two_transition_t1000_t1000"),
    ("Dual (tS=1000, tP=1200)", "balanced_2k_two_transition_t1000_t1200"),
    ("Dual (tS=1000, tP=1500)", "balanced_2k_two_transition_t1000_t1500"),
    ("Dual (tS=1000, tP=1800)", "balanced_2k_two_transition_t1000_t1800"),
    ("Dual (tS=600, tP=1500)", "balanced_2k_two_transition_t600_t1500"),
    ("Dual (tS=1400, tP=1500)", "balanced_2k_two_transition_t1400_t1500"),
]


def row(label: str, directory: str) -> dict:
    summary_path = RESULTS / directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parity = summary["parity"]
    compression = parity["compression"]
    workload = parity["segment_workload"]
    return {
        "condition": label,
        "result_directory": directory,
        "global_bytes_per_token": summary["global_bytes_per_token"],
        "compression_gini": compression["gini"],
        "compression_high_low_ratio": compression["high_low_ratio"],
        "workload_gini": workload["gini"],
        "workload_high_low_ratio": workload["high_low_ratio"],
    }


def main() -> None:
    rows = [row(label, directory) for label, directory in RUNS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table = [
        "| Condition | B/token (higher) | Compression Gini (lower) | Compression high/low (lower) | Workload Gini (lower) | Workload high/low (lower) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        table.append(
            "| {condition} | {global_bytes_per_token:.4f} | {compression_gini:.4f} | "
            "{compression_high_low_ratio:.3f} | {workload_gini:.4f} | "
            "{workload_high_low_ratio:.3f} |".format(**item)
        )
    (OUT_DIR / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))
    print(f"saved {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
