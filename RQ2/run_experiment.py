"""RQ2 exploration sweep: baselines + combination recipes x transition points.

Uses the SuperBPE truncation trick: train each subword HEAD once to the full vocab,
then (a) the head itself IS the pure-subword baseline, and (b) every transition point
is the head truncated to t merges + a continuation tail. Heads are cached on disk, so
adding more transition points later is cheap (only new tails are trained).

For each resulting tokenizer we measure, on FLORES devtest (multi-parallel):
  - compression: mean tokens/line and mean bytes/token across the 10 languages
  - parity:      disparity_ratio (worst/best), cv, gini across languages
Results stream to results/<tag>/summary.csv (one row per tokenizer, written as we go).

Run (background, unbuffered):
  ..\.venv\Scripts\python.exe -u run_experiment.py --tag explore_8k
"""

from __future__ import annotations

import sys
import csv
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipes
from engine.phased_runner import (train_head, extend_from_head, read_merges,
                                   write_merges)
from metrics import build_hf_tokenizer, measure, parity_summary

HERE = Path(__file__).resolve().parent


def log(msg):
    print(msg, flush=True)


def evaluate(merges_lines, final_stage, eval_corpora):
    """Compression + parity for one tokenizer on the FLORES devtest set."""
    tk = build_hf_tokenizer(merges_lines, stage=final_stage)
    par = parity_summary(tk, eval_corpora)
    per = par["per_language"]
    n = len(per)
    mean_tpl = sum(m["tokens_per_line"] for m in per.values()) / n
    mean_bpt = sum(m["bytes_per_token"] for m in per.values()) / n
    return {
        "mean_tokens_per_line": round(mean_tpl, 3),
        "mean_bytes_per_token": round(mean_bpt, 4),
        "disparity_ratio": round(par["disparity_ratio"], 4),
        "cv": round(par["cv"], 4),
        "gini": round(par["gini"], 4),
        "worst_language": par["worst_language"],
        "best_language": par["best_language"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="explore_8k")
    ap.add_argument("--data", default="data/train_1mb")
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--transitions", type=int, nargs="*", default=[2000, 4000, 6000])
    args = ap.parse_args()

    total = args.vocab - 256  # byte-level alphabet
    L = recipes.LANGUAGES
    inputs = [str(HERE / args.data / f"{c}.txt") for c in L]
    devs = [str(HERE / "data" / "flores" / f"{c}.dev.txt") for c in L]
    eval_corpora = {c: str(HERE / "data" / "flores" / f"{c}.devtest.txt") for c in L}
    for p in inputs + devs + list(eval_corpora.values()):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    out = HERE / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "summary.csv"
    fields = ["recipe", "transition", "total_merges", "final_stage",
              "mean_tokens_per_line", "mean_bytes_per_token",
              "disparity_ratio", "cv", "gini", "worst_language", "best_language",
              "train_seconds"]
    f_csv = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=fields)
    writer.writeheader(); f_csv.flush()

    def record(recipe, transition, merges_lines, final_stage, secs):
        row = {"recipe": recipe, "transition": transition,
               "total_merges": len(merges_lines), "final_stage": final_stage,
               "train_seconds": round(secs, 1)}
        row.update(evaluate(merges_lines, final_stage, eval_corpora))
        writer.writerow(row); f_csv.flush()
        log(f"  [done] {recipe:18s} t={str(transition):>5}  "
            f"tok/line={row['mean_tokens_per_line']:.2f}  "
            f"disparity={row['disparity_ratio']:.3f}  ({row['train_seconds']:.0f}s)")

    log(f"=== RQ2 exploration: vocab={args.vocab} (total_merges={total}), "
        f"data={args.data}, transitions={args.transitions} ===")

    # --- train the two subword heads once (cached if present) ---
    def get_head(name, rule):
        path = out / f"{name}.merges"
        if path.exists() and len(read_merges(path)) >= total:
            log(f"head '{name}': cached ({len(read_merges(path))} merges)")
            return read_merges(path), 0.0
        log(f"head '{name}': training {rule} subword to {total} merges ...")
        t0 = time.perf_counter()
        m, _ = train_head(inputs, devs, total, "sub", rule, out, name)
        dt = time.perf_counter() - t0
        log(f"head '{name}': {len(m)} merges in {dt/60:.1f} min")
        return m, dt

    head_c, t_hc = get_head("head_classical_sub", "classical")
    head_p, t_hp = get_head("head_parity_sub", "parity")

    # --- baselines (the heads themselves, evaluated as pure subword) ---
    record("classical_bpe", "", head_c, "sub", t_hc)
    record("pa_bpe", "", head_p, "sub", t_hp)

    # --- transition sweeps (all share head_classical_sub) ---
    sweep = [
        ("superbpe",          "super", "classical"),  # classical sub -> classical super
        ("bpe_then_pa_super", "super", "parity"),      # classical sub -> parity super
        ("bpe_then_pa",       "sub",   "parity"),      # classical sub -> parity sub
    ]
    for recipe, tail_stage, tail_rule in sweep:
        for t in args.transitions:
            run_name = f"{recipe}_t{t}"
            final, dt = extend_from_head(head_c, t, inputs, devs, total,
                                         tail_stage, tail_rule, out, run_name)
            record(recipe, t, read_merges(final), tail_stage, dt)

    f_csv.close()
    log(f"\nDONE -> {csv_path}")


if __name__ == "__main__":
    main()
