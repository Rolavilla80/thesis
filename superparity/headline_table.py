r"""Assemble + score a set of already-trained tokenizers into one summary.csv (so plot.py
tradeoff can draw them together). Use for the 80k headline set (trio + the two singles).

Usage:
  ..\.venv\Scripts\python.exe headline_table.py --tag headline_80k
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import config
from run import read_merges
from metrics import evaluate

# (label, single-run folder name, final stage)
DEFAULT_80K = [
    ("classical_bpe", "classical_bpe_v80000",     "sub"),
    ("pa_bpe",        "pa_bpe_v80000",            "sub"),
    ("superbpe",      "superbpe_v80000_t60000",   "super"),
    ("bpe_pa_super",  "bpe_pa_super_v80000",      "super"),
    ("bpe_super_pa",  "bpe_super_pa_v80000",      "super"),
]

FIELDS = ["recipe", "transition", "total_merges", "final_stage",
          "mean_tokens_per_line", "mean_bytes_per_token",
          "disparity_ratio", "cv", "gini", "worst_language", "best_language"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="headline_80k")
    args = ap.parse_args()
    evals = config.eval_corpora()
    out = config.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for label, folder, stage in DEFAULT_80K:
            mp = config.RESULTS / "single" / folder / f"{folder}.merges"
            if not mp.exists():
                print(f"  MISSING: {mp}"); continue
            lines = read_merges(mp)
            row = {"recipe": label, "transition": "", "total_merges": len(lines),
                   "final_stage": stage}
            row.update(evaluate(lines, stage, evals))
            w.writerow(row)
            print(f"  {label:16s} tok/line={row['mean_tokens_per_line']:.2f}  "
                  f"disp={row['disparity_ratio']:.3f}  gini={row['gini']:.3f}")
    print(f"-> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
