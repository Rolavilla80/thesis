r"""Step 0 -- the scaling test. Where do superwords start paying off for OUR languages?

Trains TWO things once, then evaluates both at many vocab checkpoints by truncating the
merge list (valid because BPE merges are sequential -> merges[:k] IS the vocab-k tokenizer):
  1. a classical SUBWORD head to --max-vocab            -> the plain-BPE curve (blue)
  2. a SuperBPE run (classical sub up to --transition, then classical super to max)
                                                          -> the SuperBPE curve
Output: results/<tag>/scaling.csv with (variant, vocab, mean_bytes_per_token,
mean_tokens_per_line, disparity_ratio). Read off the vocab where SuperBPE overtakes BPE
-> that's the final-experiment vocab. plot.py draws it.

Run (background):
  ..\.venv\Scripts\python.exe -u scaling.py --tag scaling --max-vocab 32000 --transition 8000 --data train
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import config
import run
from metrics import evaluate

FIELDS = ["variant", "transition", "vocab", "total_merges",
          "mean_bytes_per_token", "mean_tokens_per_line", "disparity_ratio", "cv", "gini"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="scaling")
    ap.add_argument("--max-vocab", type=int, default=32000)
    ap.add_argument("--transition", type=int, default=8000, help="where SuperBPE starts (vocab units)")
    ap.add_argument("--data", default="train", choices=["train", "train_1mb"])
    ap.add_argument("--checkpoints", type=int, nargs="*",
                    default=[8000, 12000, 16000, 20000, 24000, 28000, 32000])
    args = ap.parse_args()

    data_dir = config.TRAIN_FULL if args.data == "train" else config.TRAIN_1MB
    config.check_data(data_dir)
    inputs = config.train_files(data_dir)
    devs = config.dev_files()
    evals = config.eval_corpora()
    out = config.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    max_merges = args.max_vocab - config.BYTE_ALPHABET
    t_merges = args.transition - config.BYTE_ALPHABET   # transition in merge units
    ckpts = [c for c in args.checkpoints if c <= args.max_vocab]

    f_csv = open(out / "scaling.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=FIELDS); writer.writeheader(); f_csv.flush()

    def record(variant, transition, vocab, merges, stage):
        row = {"variant": variant, "transition": transition, "vocab": vocab,
               "total_merges": len(merges)}
        ev = evaluate(merges, stage, evals)
        row.update({k: ev[k] for k in ("mean_bytes_per_token", "mean_tokens_per_line",
                                        "disparity_ratio", "cv", "gini")})
        writer.writerow(row); f_csv.flush()
        print(f"  [{variant:9s}] vocab={vocab:>6}  bytes/tok={row['mean_bytes_per_token']:.3f}  "
              f"disp={row['disparity_ratio']:.3f}", flush=True)

    # 1) classical subword head to max (cached) -> BPE curve
    head_path = out / "head_classical_sub.bpe"
    if head_path.exists() and len(run.read_merges(head_path)) >= max_merges:
        print("head: cached", flush=True); head = run.read_merges(head_path)
    else:
        print(f"head: training classical subword to {max_merges} ...", flush=True)
        head, dt = run.train_head(inputs, devs, max_merges, "sub", "classical", out, "head_classical_sub")
        print(f"head: {len(head)} merges in {dt/60:.1f} min", flush=True)
    for c in ckpts:
        k = c - config.BYTE_ALPHABET
        if k <= len(head):
            record("bpe", "", c, head[:k], "sub")

    # 2) SuperBPE run: head[:t] then classical super to max -> SuperBPE curve
    print(f"superbpe: continuing from t={args.transition} in super stage to {args.max_vocab} ...", flush=True)
    final, dt = run.extend_from_head(head, t_merges, inputs, devs, max_merges,
                                     "super", "classical", out, f"superbpe_t{args.transition}")
    sup = run.read_merges(final)
    print(f"superbpe: {len(sup)} merges in {dt/60:.1f} min", flush=True)
    for c in ckpts:
        if c < args.transition:
            continue
        k = c - config.BYTE_ALPHABET
        if k <= len(sup):
            record("superbpe", args.transition, c, sup[:k], "super")

    f_csv.close()
    print(f"\nDONE -> {out / 'scaling.csv'}", flush=True)


if __name__ == "__main__":
    main()
