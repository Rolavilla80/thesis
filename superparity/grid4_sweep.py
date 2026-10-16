r"""4-PHASE GRID: BPE -> SuperBPE -> PA-super -> SuperBPE   (the advisor's 4-phase reading)

Three transitions: t1 (sub->super, fixed on the plateau), t2 (classical->parity),
t3 (parity->classical again). We sweep (t2, t3) as a small grid.

REGISTERED PREDICTION (from the ordering experiment, F1 "fairness has recency"):
because this recipe ENDS on a classical phase, each cell's fairness should be DILUTED
relative to the 3-phase A-frontier at the same parity budget — dominated in the fair
region, possibly competitive in the compressed region. This run tests that prediction.

CHEAP because everything before the last phase is already trained:
    prefix = spine[:t2]                (classical head+super, explore_8k spine)
           + parity_tail(t2)[:t3-t2]   (the pareto_8k parity tails, trained to full)
    then train only the final classical-super tail (total - t3 merges) per cell.

The plot overlays the 3-phase A-frontier (from results/pareto_8k/summary.csv) so each
cell can be judged against the incumbent menu at a glance.

Run (from superparity/):
  ..\.venv\Scripts\python.exe grid4_sweep.py --tag grid4_8k --vocab 8000 ^
      --spine results\explore_8k\superbpe_t2000.merges --parity-dir results\pareto_8k ^
      --t1 2000 --t2 2904 3872 4840 --t3 5808 6776 7260 --parallel 6
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

FIELDS = ["recipe", "t1", "t2", "t3", "t1_frac", "t2_frac", "t3_frac", "parity_frac",
          "total_merges", "final_stage",
          "mean_tokens_per_line", "mean_bytes_per_token",
          "disparity_ratio", "cv", "gini", "worst_language", "best_language",
          "train_seconds"]


def read_ref_frontier() -> list[tuple[float, float, float]]:
    """(tok/line, disparity, t2_frac) of the 3-phase A-frontier, for the overlay."""
    path = config.RESULTS / "pareto_8k" / "summary.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [(float(r["mean_tokens_per_line"]), float(r["disparity_ratio"]),
                 float(r["t2_frac"])) for r in csv.DictReader(f)]


def plot_grid(rows: list[dict], ref, out_dir: Path, vocab: int, refresh_s: int = 60):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    if ref:
        rx = [p[0] for p in ref]
        ry = [p[1] for p in ref]
        ax.scatter(rx, ry, marker="x", s=60, color="0.45", zorder=2,
                   label="3-phase A-frontier (ends on parity)")
    xs = [float(r["mean_tokens_per_line"]) for r in rows]
    ys = [float(r["disparity_ratio"]) for r in rows]
    cs = [float(r["parity_frac"]) for r in rows]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=95, zorder=3,
                    edgecolors="white", linewidths=0.6)
    fig.colorbar(sc, ax=ax, label="parity budget (fraction of total merges)")
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(f"({float(r['t2_frac']):.2f},{float(r['t3_frac']):.2f})", (x, y),
                    fontsize=6.5, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("mean tokens / line  (lower = better compression)")
    ax.set_ylabel("disparity ratio  (lower = fairer)")
    ax.set_title(f"4-phase grid @ {vocab//1000}k — BPE→Super→PA→Super, labels = (t2, t3)\n"
                 "prediction: classical finish dilutes fairness vs the A-frontier",
                 fontsize=10)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "grid.png", dpi=140); plt.close(fig)
    (out_dir / "grid.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="{refresh_s}"><title>grid4</title></head>'
        f'<body style="font-family:system-ui;margin:24px">'
        f'<h2>4-phase grid — vocab {vocab}</h2><p>updated {time.strftime("%H:%M:%S")}</p>'
        f'<img src="grid.png?ts={int(time.time())}" style="max-width:100%">'
        f'</body></html>', encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--data", default="train_1mb", choices=["train", "train_1mb", "train_big"])
    ap.add_argument("--spine", required=True,
                    help="classical spine (BPE head + classical-super to full length)")
    ap.add_argument("--parity-dir", required=True,
                    help="dir holding parity_t2<N>.tail.bpe files (e.g. results/pareto_8k)")
    ap.add_argument("--t1", type=int, required=True)
    ap.add_argument("--t2", type=int, nargs="+", required=True,
                    help="where parity starts (must have a parity_t2<N>.tail.bpe)")
    ap.add_argument("--t3", type=int, nargs="+", required=True,
                    help="where parity ENDS and classical resumes (t3 > t2)")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--no-beep", action="store_true")
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
        raise SystemExit(f"spine has {len(spine)} < {total} merges")
    ptails = {}
    for t2 in sorted(set(args.t2)):
        p = Path(args.parity_dir) / f"parity_t2{t2}.tail.bpe"
        if not p.exists():
            raise SystemExit(f"missing parity tail: {p}")
        ptails[t2] = run.read_merges(p)

    cells = [(t2, t3) for t2 in sorted(set(args.t2)) for t3 in sorted(set(args.t3))
             if args.t1 < t2 < t3 <= total and (t3 - t2) <= len(ptails[t2])]
    ref = read_ref_frontier()
    print(f"=== grid4 '{args.tag}': vocab={args.vocab} (total={total}), t1={args.t1}, "
          f"parallel={args.parallel} ===", flush=True)
    print(f"recipe: BPE({args.t1}) -> Super(to t2) -> PA-super(to t3) -> Super(rest)",
          flush=True)
    print(f"{len(cells)} cells; final classical tails = "
          f"{sorted({total - t3 for _, t3 in cells}, reverse=True)}\n", flush=True)

    f = open(out / "summary.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader(); f.flush()
    lock = threading.Lock()
    rows_done: list[dict] = []
    n_done = [0]

    def train_cell(cell):
        t2, t3 = cell
        prefix = spine[:t2] + ptails[t2][:t3 - t2]
        if t3 >= total:
            return t2, t3, prefix, 0.0
        pre = out / f"g4_t2{t2}_t3{t3}.preload.bpe"
        run.write_merges(pre, prefix, header=False)
        tail_out = out / f"g4_t2{t2}_t3{t3}.tail.bpe"
        dt = run.run_phase(inputs, devs, total - t3, "super", "classical", tail_out, pre)
        merges = prefix + run.read_merges(tail_out)
        run.write_merges(out / f"g4_t2{t2}_t3{t3}.merges", merges, header=True)
        return t2, t3, merges, dt

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(train_cell, c) for c in cells]
        for fut in as_completed(futures):
            t2, t3, merges, dt = fut.result()
            row = {"recipe": "bpe_super_pa_super", "t1": args.t1, "t2": t2, "t3": t3,
                   "t1_frac": round(args.t1 / total, 3), "t2_frac": round(t2 / total, 3),
                   "t3_frac": round(t3 / total, 3),
                   "parity_frac": round((t3 - t2) / total, 3),
                   "total_merges": len(merges), "final_stage": "super",
                   "train_seconds": round(dt, 1)}
            row.update(evaluate(merges, "super", evals))
            with lock:
                writer.writerow(row); f.flush()
                rows_done.append(row)
                n_done[0] += 1
                print(f"  [done {n_done[0]}/{len(cells)}] t2={t2} t3={t3} "
                      f"(parity {row['parity_frac']:.2f})  "
                      f"tok/line={row['mean_tokens_per_line']:.2f}  "
                      f"disp={row['disparity_ratio']:.3f}  ({dt/60:.1f} min)", flush=True)
                plot_grid(rows_done, ref, out, args.vocab)
                if not args.no_beep:
                    _beep(2)
    f.close()
    print(f"\nDONE -> {out / 'summary.csv'}  +  grid.png", flush=True)
    if not args.no_beep:
        _beep_finish()


if __name__ == "__main__":
    main()
