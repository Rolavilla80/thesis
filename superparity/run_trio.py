r"""Headline trio: train classical_bpe + superbpe + bpe_pa_super at ONE vocab, IN PARALLEL
across cores, then score all three into one table. The cheap way to get a comparison table
at a big vocab without the full 18-recipe sweep (which would take weeks at 80k+).

Each recipe runs as its own `run.py` process (so they use separate cores for their merge
loops). When all finish, we evaluate each merge list and write one summary.csv.

RAM note: superbpe and bpe_pa_super each have a heavy superword phase; at big vocab running
them together uses a lot of RAM. If it swaps, lower --max-parallel (e.g. 2) or --vocab.

Usage:
  ..\.venv\Scripts\python.exe -u run_trio.py --vocab 80000 --data train_1mb --workers 4
"""

from __future__ import annotations

import csv
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import config
import recipes
from run import read_merges
from metrics import evaluate

TRIO = ["classical_bpe", "superbpe", "bpe_pa_super"]   # baseline / compression-only / combo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=80000)
    ap.add_argument("--data", default="train_1mb", choices=["train", "train_1mb"])
    ap.add_argument("--transition", type=int, default=None,
                    help="superbpe transition point (default: vocab//3)")
    ap.add_argument("--workers", type=int, default=1,
                    help="engine preprocessing cores per run; keep 1 (engine's >1 path is buggy)")
    ap.add_argument("--max-parallel", type=int, default=3, help="how many runs at once (lower if RAM tight)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    data_dir = config.TRAIN_FULL if args.data == "train" else config.TRAIN_1MB
    config.check_data(data_dir)
    transition = args.transition or args.vocab // 3
    total = args.vocab - config.BYTE_ALPHABET
    tag = args.tag or f"trio_v{args.vocab}"
    out = config.RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)

    # build a job per recipe
    jobs = []
    for recipe in TRIO:
        needs_t = recipes.RECIPES[recipe][1]
        name = f"{recipe}_v{args.vocab}" + (f"_t{transition}" if needs_t else "")
        phases = recipes.build(recipe, total, transition if needs_t else None)
        cmd = [config.PYTHON, "-u", str(config.HERE / "run.py"), recipe,
               "--vocab", str(args.vocab), "--data", args.data,
               "--workers", str(args.workers), "--quiet", "--name", name]
        if needs_t:
            cmd += ["--transition", str(transition)]
        merges_path = config.RESULTS / "single" / name / f"{name}.merges"
        jobs.append({"recipe": recipe, "name": name, "stage": recipes.final_stage(phases),
                     "merges": merges_path, "cmd": cmd, "log": out / f"{name}.log"})

    print(f"=== trio @ vocab {args.vocab} ({args.data}), {args.max_parallel} in parallel, "
          f"{args.workers} workers each ===", flush=True)
    for j in jobs:
        print(f"  - {j['name']}  -> log {j['log'].name}", flush=True)

    def run_one(j):
        t0 = time.perf_counter()
        with open(j["log"], "w", encoding="utf-8") as lf:
            rc = subprocess.run(j["cmd"], stdout=lf, stderr=subprocess.STDOUT).returncode
        j["seconds"] = time.perf_counter() - t0
        j["rc"] = rc
        print(f"  [{'ok' if rc == 0 else 'FAIL'}] {j['name']}  ({j['seconds']/60:.1f} min)", flush=True)
        return j

    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        done = list(ex.map(run_one, jobs))

    # evaluate
    evals = config.eval_corpora()
    fields = ["recipe", "vocab", "total_merges", "final_stage", "mean_tokens_per_line",
              "mean_bytes_per_token", "disparity_ratio", "cv", "gini",
              "worst_language", "best_language", "train_seconds"]
    csv_path = out / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for j in done:
            if j["rc"] != 0 or not j["merges"].exists():
                print(f"  [skip eval] {j['name']} (run failed or no merges; see {j['log'].name})", flush=True)
                continue
            lines = read_merges(j["merges"])
            row = {"recipe": j["recipe"], "vocab": args.vocab, "total_merges": len(lines),
                   "final_stage": j["stage"], "train_seconds": round(j["seconds"], 1)}
            row.update(evaluate(lines, j["stage"], evals))
            w.writerow(row)
            print(f"  [scored] {j['recipe']:14s} tok/line={row['mean_tokens_per_line']:.2f} "
                  f"disp={row['disparity_ratio']:.3f}", flush=True)
    print(f"\nDONE -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
