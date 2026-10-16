r"""RUNG 1 — the transition-point sweep (the advisor's new direction, 2026-06-26).

We stop hand-picking the phase-split fractions of `bpe_super_pa` and instead SWEEP its
two transition points, mapping a 2D surface of BOTH objectives. This is the SuperBPE
Fig-2 idea, but richer: two transitions and two objectives at once.

`bpe_super_pa` = three phases, two transitions (t1 < t2 <= total):
    [0 , t1)    sub  , classical   (plain BPE)
    [t1, t2)    super, classical   (SuperBPE)
    [t2, total) super, parity      (PA-super)
    t1 = sub->super,  t2 = classical->parity.  The advisor's default = t1=1/4, t2=1/2.

NESTED TRUNCATION (makes the 2D map cheap; BPE merges are greedy/sequential):
  * train the classical-sub HEAD once to max(t1)         -> slice for any t1.
  * for each t1, train the super-classical TAIL once to   -> slice for any t2.
      max(t2)-t1 merges (preload = head[:t1]).
  * only the super-PARITY tail is per-cell (preload = head[:t1] + superclassical[:t2-t1]).
Cost = 1 head + |t1| super-classical tails + |cells| super-parity tails.

Corners reconnect to known recipes (sanity):
    t2 == total  -> no parity phase   = superbpe (classical sub->super)
    t2 == t1     -> no classical-super = bpe_then_pa_super
    t1 small, t2=1/2 etc.            = the advisor's hand-picked bpe_super_pa

Rows stream to results/<tag>/surface.csv. Plot: `python plot.py surface --csv ...`.

Run (background, unbuffered):
  ..\.venv\Scripts\python.exe -u transition_sweep.py --tag rung1_8k --vocab 8000 --data train_1mb
"""

from __future__ import annotations

import csv
import time
import argparse
from pathlib import Path

import config
import run
from metrics import evaluate

FIELDS = ["t1", "t2", "t1_frac", "t2_frac", "total_merges", "final_stage",
          "mean_tokens_per_line", "mean_bytes_per_token",
          "disparity_ratio", "cv", "gini", "worst_language", "best_language",
          "kind", "train_seconds"]


def _grid(fracs: list[float], total: int, hi: int) -> list[int]:
    """Fractions of `total` -> sorted distinct merge counts in (0, hi]."""
    vals = sorted({round(f * total) for f in fracs})
    return [v for v in vals if 0 < v <= hi]


def main():
    ap = argparse.ArgumentParser(description="2D transition sweep of bpe_super_pa.")
    ap.add_argument("--tag", default="rung1_8k")
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--data", default="train_1mb", choices=["train", "train_1mb"])
    ap.add_argument("--t1-fracs", type=float, nargs="*",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help="sub->super transition, as fractions of total merges")
    ap.add_argument("--t2-fracs", type=float, nargs="*",
                    default=[0.2, 0.4, 0.6, 0.8, 1.0],
                    help="classical->parity transition, as fractions of total merges")
    ap.add_argument("--progress", action="store_true",
                    help="stream the engine's per-phase progress bar")
    ap.add_argument("--workers", type=int, default=1,
                    help="engine preprocessing cores; keep 1 (engine's >1 path is buggy)")
    args = ap.parse_args()
    run.SHOW_PROGRESS = args.progress
    run.NUM_WORKERS = args.workers

    data_dir = config.TRAIN_FULL if args.data == "train" else config.TRAIN_1MB
    config.check_data(data_dir)
    total = args.vocab - config.BYTE_ALPHABET
    inputs = config.train_files(data_dir)
    devs = config.dev_files()
    evals = config.eval_corpora()
    out = config.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    t1_vals = _grid(args.t1_fracs, total, hi=total - 1)           # need room for >=1 super
    t2_vals = _grid(args.t2_fracs, total, hi=total)               # t2==total allowed (=superbpe)
    max_t1 = max(t1_vals)
    max_t2 = max(t2_vals)
    cells = [(t1, t2) for t1 in t1_vals for t2 in t2_vals if t2 >= t1]

    print(f"=== rung1 transition sweep '{args.tag}': vocab={args.vocab} (total={total}), "
          f"data={args.data} ===", flush=True)
    print(f"t1 (sub->super)      = {t1_vals}", flush=True)
    print(f"t2 (classical->parity) = {t2_vals}", flush=True)
    print(f"{len(cells)} cells (upper triangle t2>=t1)\n", flush=True)

    f_csv = open(out / "surface.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=FIELDS)
    writer.writeheader(); f_csv.flush()

    def record(t1, t2, merges, secs, kind):
        row = {"t1": t1, "t2": t2,
                "t1_frac": round(t1 / total, 3), "t2_frac": round(t2 / total, 3),
                "total_merges": len(merges), "final_stage": "super",
                "kind": kind, "train_seconds": round(secs, 1)}
        row.update(evaluate(merges, "super", evals))
        writer.writerow(row); f_csv.flush()
        print(f"  [done] t1={t1:5d} t2={t2:5d} ({row['t1_frac']:.2f}/{row['t2_frac']:.2f})  "
              f"tok/line={row['mean_tokens_per_line']:.2f}  "
              f"disp={row['disparity_ratio']:.3f}  gini={row['gini']:.3f}  "
              f"({row['train_seconds']:.0f}s)", flush=True)

    # --- classical-sub HEAD, trained once (cached) ---
    head_path = out / "head_classical_sub.bpe"
    if head_path.exists() and len(run.read_merges(head_path)) >= max_t1:
        head = run.read_merges(head_path)
        print(f"head: cached ({len(head)} merges)", flush=True)
    else:
        print(f"head: training classical-sub to {max_t1} ...", flush=True)
        head, dt = run.train_head(inputs, devs, max_t1, "sub", "classical",
                                  out, "head_classical_sub")
        print(f"head: {len(head)} merges in {dt/60:.1f} min", flush=True)

    # --- per t1: super-classical TAIL once (cached), then per-cell super-parity ---
    for t1 in t1_vals:
        prefix = head[:t1]                                # classical-sub phase
        sc_len = max_t2 - t1                              # super-classical merges needed
        sc_path = out / f"superclassical_t1{t1}.bpe"
        if sc_len > 0 and sc_path.exists() and len(run.read_merges(sc_path)) >= sc_len:
            sc = run.read_merges(sc_path)
            print(f"\nt1={t1}: super-classical tail cached ({len(sc)} merges)", flush=True)
        elif sc_len > 0:
            print(f"\nt1={t1}: super-classical tail, {sc_len} merges ...", flush=True)
            pre = out / f"_pre_sc_t1{t1}.bpe"
            run.write_merges(pre, prefix, header=False)
            dt = run.run_phase(inputs, devs, sc_len, "super", "classical", sc_path, pre)
            sc = run.read_merges(sc_path)
            print(f"t1={t1}: super-classical {len(sc)} merges in {dt/60:.1f} min", flush=True)
        else:
            sc = []

        for t2 in t2_vals:
            if t2 < t1:
                continue
            classical_super = sc[:t2 - t1]                # super-classical portion (SuperBPE)
            base = prefix + classical_super               # everything before the parity phase
            par_len = total - t2                          # super-parity merges
            if par_len <= 0:
                # t2 == total: no parity phase -> this cell IS superbpe
                record(t1, t2, base, 0.0, kind="superbpe_edge")
                continue
            pre = out / f"_pre_par_t1{t1}_t2{t2}.bpe"
            run.write_merges(pre, base, header=False)
            tail_path = out / f"parity_t1{t1}_t2{t2}.bpe"
            dt = run.run_phase(inputs, devs, par_len, "super", "parity", tail_path, pre)
            merges = base + run.read_merges(tail_path)
            kind = "bpe_then_pa_super_edge" if t2 == t1 else "interior"
            record(t1, t2, merges, dt, kind=kind)
            pre.unlink(missing_ok=True)                   # tidy the per-cell preload

    f_csv.close()
    print(f"\nDONE -> {out / 'surface.csv'}", flush=True)


if __name__ == "__main__":
    main()
