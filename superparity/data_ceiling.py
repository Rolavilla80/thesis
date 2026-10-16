r"""Data-ceiling probe: can the current data sustain a big vocabulary?

A BPE merge needs its pair to occur >= 2 times. As the vocabulary grows, the most-frequent
pair's count falls; when it reaches 2 the engine STOPS EARLY -- that's the data ceiling.
This probe trains a classical *subword* head while logging each merge's frequency, then
reports the decay so you can judge headroom BEFORE committing to a long big-vocab run.

Reading the result:
  - if it stops before --target  -> that merge count IS your subword ceiling on this data.
  - last-merge frequency >> 2     -> lots of headroom (data can go much higher).
  - last-merge frequency near 2   -> you are at the limit; get more data.
Note: the SUPERWORD phase is MORE data-hungry (repeated multi-word phrases are rarer), so
this subword ceiling is an optimistic upper bound. Use --probe-super to also probe it.

The trained head is saved and reusable as phase-1 of the big run (truncate to the phase size).

Usage:
  ..\.venv\Scripts\python.exe -u data_ceiling.py --data train --target 60000
  ..\.venv\Scripts\python.exe -u data_ceiling.py --data train_1mb --target 40000 --probe-super
"""

from __future__ import annotations

import csv
import sys
import argparse
import subprocess
from pathlib import Path

import config


def _train(stage_pretok, n, out_merges, log_path, preload=None):
    cmd = [config.PYTHON, str(config.ENGINE),
           "-i", *config.train_files(_DATA),
           "-d", *config.dev_files(),
           "--pretokenize", *stage_pretok,
           "--symbols", str(n), "--global-merges", str(n),
           "--output", str(out_merges), "--merge-log", str(log_path)]
    if preload:
        cmd += ["--preload", str(preload)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("ENGINE FAILED:\n" + proc.stderr[-2000:]); sys.exit(1)


def _counts(log_path):
    """Total frequency of each merged pair = sum of its per-language frequency_vector.
    Training stops when this can no longer reach min_frequency (2) -> the data ceiling."""
    out = []
    for r in csv.DictReader(open(log_path, encoding="utf-8")):
        fv = r.get("frequency_vector", "")
        if fv:
            try:
                out.append(sum(int(x) for x in fv.split(";")))
            except ValueError:
                pass
    return out


def _report(label, counts, target):
    achieved = len(counts)
    print(f"\n=== {label} ===")
    print(f"achieved merges: {achieved}  (target {target})")
    if achieved < target:
        print(f"** CEILING: data exhausted at {achieved} merges (pairs stopped repeating) **")
    else:
        print("** reached target without exhausting data -- could go higher **")
    print("frequency of the top pair as the vocab grows:")
    for k in (1000, 5000, 10000, 20000, 30000, 40000, 60000, 80000, 120000):
        if k <= achieved:
            print(f"   at {k:7d} merges: top-pair frequency = {counts[k-1]}")
    if counts:
        last = counts[-1]
        verdict = ("LOTS of headroom" if last >= 20 else
                   "some headroom" if last >= 6 else
                   "NEAR THE LIMIT -- get more data")
        print(f"   last merge frequency = {last}  (2 = floor)  -> {verdict}")


def main():
    global _DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train", choices=["train", "train_1mb"])
    ap.add_argument("--target", type=int, default=60000, help="subword merges to attempt")
    ap.add_argument("--super-target", type=int, default=20000,
                    help="extra superword merges to attempt when --probe-super")
    ap.add_argument("--probe-super", action="store_true",
                    help="also continue into a classical superword phase (slower, but the binding limit)")
    ap.add_argument("--tag", default="ceiling")
    args = ap.parse_args()

    _DATA = config.TRAIN_FULL if args.data == "train" else config.TRAIN_1MB
    config.check_data(_DATA)
    out = config.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    sub_merges = out / f"{args.data}_sub.bpe"
    sub_log = out / f"{args.data}_sub_mergelog.csv"
    print(f"probing SUBWORD ceiling on '{args.data}' up to {args.target} merges "
          f"(stops early if data runs out) ...", flush=True)
    _train(["whitespace", "bytelevel"], args.target, sub_merges, sub_log)
    sub_counts = _counts(sub_log)
    _report(f"SUBWORD ceiling ({args.data})", sub_counts, args.target)

    if args.probe_super:
        # continue from the subword head into superwords
        preload = out / f"{args.data}_sub_preload.bpe"
        lines = [l for l in sub_merges.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#version")]
        preload.write_text("\n".join(lines) + "\n", encoding="utf-8")
        sup_merges = out / f"{args.data}_super.bpe"
        sup_log = out / f"{args.data}_super_mergelog.csv"
        print(f"\nprobing SUPERWORD ceiling: +{args.super_target} superword merges ...", flush=True)
        _train(["bytelevel"], args.super_target, sup_merges, sup_log, preload=preload)
        _report(f"SUPERWORD ceiling ({args.data})", _counts(sup_log), args.super_target)

    print("\nDONE. The subword head is reusable as phase-1 of the big run.")


if __name__ == "__main__":
    main()
