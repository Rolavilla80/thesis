r"""Exploration sweep: baselines + recipes x transition points at one vocab size.

Speed trick (SuperBPE's own): train each subword HEAD once to full length, then every
transition point is head[:t] + a continuation tail (run.extend_from_head). Heads are
cached on disk, so re-running / adding transitions is cheap.

For each tokenizer we measure, on FLORES devtest (multi-parallel):
  compression: mean tokens/line, mean bytes/token   parity: disparity_ratio, cv, gini
Rows stream to results/<tag>/summary.csv as they finish.

Run (background, unbuffered):
  ..\.venv\Scripts\python.exe -u sweep.py --tag explore_8k --vocab 8000 --data train_1mb
"""

from __future__ import annotations

import csv
import time
import argparse
from pathlib import Path

import config
import recipes
import run
from metrics import evaluate

FIELDS = ["recipe", "transition", "total_merges", "final_stage",
          "mean_tokens_per_line", "mean_bytes_per_token",
          "disparity_ratio", "cv", "gini", "worst_language", "best_language",
          "train_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="explore_8k")
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--data", default="train_1mb", choices=["train", "train_1mb"])
    ap.add_argument("--transitions", type=int, nargs="*", default=[2000, 4000, 6000])
    ap.add_argument("--multiphase", action="store_true",
                    help="also run the 3-/4-phase recipes (slower; trained fresh)")
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

    f_csv = open(out / "summary.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=FIELDS)
    writer.writeheader(); f_csv.flush()

    def record(recipe, transition, merges, final_stage, secs):
        row = {"recipe": recipe, "transition": transition, "total_merges": len(merges),
               "final_stage": final_stage, "train_seconds": round(secs, 1)}
        row.update(evaluate(merges, final_stage, evals))
        writer.writerow(row); f_csv.flush()
        print(f"  [done] {recipe:18s} t={str(transition):>6}  "
              f"tok/line={row['mean_tokens_per_line']:.2f}  "
              f"disp={row['disparity_ratio']:.3f}  ({row['train_seconds']:.0f}s)", flush=True)

    print(f"=== sweep '{args.tag}': vocab={args.vocab} (total={total}), data={args.data}, "
          f"transitions={args.transitions} ===", flush=True)

    # --- the two subword heads, trained once (cached) ---
    def get_head(name, rule):
        path = out / f"{name}.bpe"
        if path.exists() and len(run.read_merges(path)) >= total:
            print(f"head '{name}': cached", flush=True)
            return run.read_merges(path), 0.0
        print(f"head '{name}': training {rule} subword to {total} ...", flush=True)
        m, dt = run.train_head(inputs, devs, total, "sub", rule, out, name)
        print(f"head '{name}': {len(m)} merges in {dt/60:.1f} min", flush=True)
        return m, dt

    head_c, t_hc = get_head("head_classical_sub", "classical")
    head_p, t_hp = get_head("head_parity_sub", "parity")

    # --- baselines (the heads themselves, as pure subword) ---
    record("classical_bpe", "", head_c, "sub", t_hc)
    record("pa_bpe", "", head_p, "sub", t_hp)

    # --- single-transition recipes from the classical head ---
    for recipe, tail_stage, tail_rule in [
            ("superbpe",          "super", "classical"),
            ("bpe_then_pa_super", "super", "parity"),
            ("bpe_then_pa",       "sub",   "parity")]:
        for t in args.transitions:
            final, dt = run.extend_from_head(head_c, t, inputs, devs, total,
                                             tail_stage, tail_rule, out, f"{recipe}_t{t}")
            record(recipe, t, run.read_merges(final), tail_stage, dt)

    # --- from the parity head ---
    for t in args.transitions:
        final, dt = run.extend_from_head(head_p, t, inputs, devs, total,
                                         "super", "classical", out, f"pa_then_super_t{t}")
        record("pa_then_super", t, run.read_merges(final), "super", dt)

    # --- pa_super baseline (parity super from scratch) ---
    final = run.run_recipe(recipes.pa_super(total), inputs, devs, out, "pa_super", verbose=False)
    record("pa_super", "", run.read_merges(final), "super", 0.0)

    # --- optional multi-phase recipes (trained fresh) ---
    if args.multiphase:
        for name in ["bpe_pa_super", "bpe_super_pa", "bpe_pa_super_pa"]:
            phases = recipes.build(name, total)
            t0 = time.perf_counter()
            final = run.run_recipe(phases, inputs, devs, out, name, verbose=False)
            record(name, "", run.read_merges(final), recipes.final_stage(phases),
                   time.perf_counter() - t0)

    f_csv.close()
    print(f"\nDONE -> {out / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
