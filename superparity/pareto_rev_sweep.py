r"""ORDERING TEST: the mirror image of pareto_sweep.py.

pareto_sweep  : BPE -> SuperBPE -> PA-super   (parity LAST,   sweep when parity STARTS)
this script   : BPE -> PA-super -> SuperBPE   (parity MIDDLE, sweep when parity ENDS)

Both sweeps connect the SAME two corner recipes:
    all-classical end  = superbpe              (t2 = t1 here / t2 = total there)
    all-parity end     = bpe_then_pa_super     (t2 = total here / t2 = t1 there)
so comparing the two frontiers holds ingredients AND budget fixed and varies only the
ORDER of the phases. Question it answers: does finishing on a classical-super phase
"undo" parity's fairness gains (because classical goes back to serving the most
frequent language), or is only the parity BUDGET what matters?

Cheap for the same reason as pareto_sweep, with the roles swapped: the SPINE here is a
parity continuation (BPE head + PA-super all the way to total) which is sliceable at
any t2; only the trailing CLASSICAL-super tail is trained per point.
Ready-made spine on disk: results/pareto_8k/parity_t22000.merges
    = BPE(2000) + PA-super(5744)  -- exactly what this recipe needs at 8k.

Run (from superparity/):
  ..\.venv\Scripts\python.exe pareto_rev_sweep.py --tag pareto_8k_rev --vocab 8000 ^
      --spine results\pareto_8k\parity_t22000.merges --t1 2000 ^
      --t2 2904 3872 4840 5808 6776 7744 --parallel 6

t2 here = the merge index where PARITY ENDS and classical-super resumes.
t2 = total -> no classical phase at all (free eval: the spine itself).
"""

from __future__ import annotations

import csv
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import run
from metrics import evaluate
from fig2_sweep import _beep, _beep_finish

FIELDS = ["recipe", "t1", "t2", "t1_frac", "t2_frac", "total_merges", "final_stage",
          "mean_tokens_per_line", "mean_bytes_per_token",
          "disparity_ratio", "cv", "gini", "worst_language", "best_language",
          "train_seconds"]


def plot_frontier(rows: list[dict], out_dir: Path, vocab: int, t1: int,
                  refresh_s: int = 60) -> None:
    xs = [float(r["mean_tokens_per_line"]) for r in rows]
    ys = [float(r["disparity_ratio"]) for r in rows]
    fr = [float(r["t2_frac"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, c=fr, cmap="plasma", s=90, zorder=3,
                    edgecolors="white", linewidths=0.6)
    fig.colorbar(sc, ax=ax, label="t2 (fraction) — where parity ENDS; large = more parity")
    for x, y, f in zip(xs, ys, fr):
        ax.annotate(f"t2={f:.2f}", (x, y), fontsize=7.5,
                    xytext=(6, 4), textcoords="offset points")
    pts = sorted(zip(xs, ys))
    front, best_y = [], float("inf")
    for x, y in pts:
        if y < best_y:
            front.append((x, y)); best_y = y
    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front], "r--", lw=1.2,
                alpha=0.7, zorder=2, label="Pareto front")
        ax.legend(fontsize=8)
    ax.set_xlabel("mean tokens / line  (lower = better compression)")
    ax.set_ylabel("disparity ratio  (lower = fairer)")
    ax.set_title(f"REVERSED order @ vocab {vocab//1000}k — BPE→PA-super→SuperBPE\n"
                 f"(t1={t1} fixed; sweeping t2 = where parity ends)", fontsize=10)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "frontier.png", dpi=140); plt.close(fig)
    (out_dir / "frontier.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="{refresh_s}"><title>frontier (rev)</title>'
        f'</head><body style="font-family:system-ui;margin:24px">'
        f'<h2>Reversed-order frontier — vocab {vocab} (t1={t1})</h2>'
        f'<p>updated {time.strftime("%H:%M:%S")}</p>'
        f'<img src="frontier.png?ts={int(time.time())}" style="max-width:100%">'
        f'</body></html>', encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--data", default="train_1mb", choices=["train", "train_1mb", "train_big"])
    ap.add_argument("--spine", required=True,
                    help="PARITY spine: BPE head + PA-super to full length "
                         "(e.g. results/pareto_8k/parity_t22000.merges)")
    ap.add_argument("--t1", type=int, required=True,
                    help="the spine's sub->super transition (also where parity began)")
    ap.add_argument("--t2", type=int, nargs="+", required=True,
                    help="where parity ENDS and classical-super resumes")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--no-beep", action="store_true")
    ap.add_argument("--recipe-name", default="bpe_pasuper_super",
                    help="label written to the CSV. Use 'bpe_pasuper_super' for a "
                         "parity-SUPER spine (BPE->PA-super->SuperBPE) or "
                         "'bpe_pa_super' for a parity-SUB spine (BPE->PA-sub->SuperBPE)")
    args = ap.parse_args()
    run.SHOW_PROGRESS = False
    run.NUM_WORKERS = 1

    data_dir = config.DATA_DIRS[args.data]
    config.check_data(data_dir)
    total = args.vocab - config.BYTE_ALPHABET
    inputs = config.train_files(data_dir)
    devs = config.dev_files()
    evals = config.eval_corpora()
    out = config.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    spine = run.read_merges(args.spine)
    if len(spine) < total:
        raise SystemExit(f"spine has {len(spine)} merges < total {total} needed")
    t2s = sorted(t for t in args.t2 if args.t1 <= t <= total)
    print(f"=== pareto REVERSED '{args.tag}': vocab={args.vocab} (total={total}), "
          f"t1={args.t1}, data={args.data}, parallel={args.parallel} ===", flush=True)
    print(f"recipe: BPE({args.t1}) -> PA-super(until t2) -> SuperBPE(rest)", flush=True)
    print(f"spine (parity): {args.spine} ({len(spine)} merges)", flush=True)
    print(f"t2 = {t2s}  (fracs {[round(t/total, 3) for t in t2s]})", flush=True)
    print(f"classical-super tail lengths = {[total - t for t in t2s]}\n", flush=True)

    f = open(out / "summary.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader(); f.flush()
    lock = threading.Lock()
    rows_done: list[dict] = []
    n_done = [0]

    def train_point(t2: int):
        if t2 >= total:                       # no classical phase: the spine itself
            return t2, spine[:total], 0.0
        pre = out / f"rev_t2{t2}.preload.bpe"
        run.write_merges(pre, spine[:t2], header=False)
        tail_out = out / f"rev_t2{t2}.tail.bpe"
        # THE MIRROR: classical tail after a parity middle
        dt = run.run_phase(inputs, devs, total - t2, "super", "classical", tail_out, pre)
        merges = spine[:t2] + run.read_merges(tail_out)
        run.write_merges(out / f"rev_t2{t2}.merges", merges, header=True)
        return t2, merges, dt

    print(f"training {len(t2s)} points, {args.parallel} at a time ...\n", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(train_point, t) for t in t2s]
        for fut in as_completed(futures):
            t2, merges, dt = fut.result()
            row = {"recipe": args.recipe_name, "t1": args.t1, "t2": t2,
                   "t1_frac": round(args.t1 / total, 3), "t2_frac": round(t2 / total, 3),
                   "total_merges": len(merges), "final_stage": "super",
                   "train_seconds": round(dt, 1)}
            row.update(evaluate(merges, "super", evals))
            with lock:
                writer.writerow(row); f.flush()
                rows_done.append(row)
                n_done[0] += 1
                print(f"  [done {n_done[0]}/{len(t2s)}] t2={t2:6d} ({t2/total:.2f})  "
                      f"tok/line={row['mean_tokens_per_line']:.2f}  "
                      f"disp={row['disparity_ratio']:.3f}  gini={row['gini']:.3f}  "
                      f"({dt/60:.1f} min)", flush=True)
                plot_frontier(rows_done, out, args.vocab, args.t1)
                if not args.no_beep:
                    _beep(2)
    f.close()
    print(f"\nDONE -> {out / 'summary.csv'}  +  frontier.png", flush=True)
    if not args.no_beep:
        _beep_finish()


if __name__ == "__main__":
    main()
