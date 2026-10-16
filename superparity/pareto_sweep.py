r"""The PARETO FRONTIER: sweep t2 (when parity turns ON) at fixed t1, one vocab.

Recipe = bpe_super_pa with t1 frozen on the compression plateau (the fig2 curves
proved t1 barely matters), sweeping t2:
    [0,t1) classical-sub | [t1,t2) classical-super | [t2,total) PARITY-super
Small t2 = big parity budget = fairest.  t2 = total = plain superbpe (no parity).
Each setting -> one dot in (compression, fairness) space; the non-dominated dots
form the Pareto frontier = the menu of best trade-offs (the professor's deliverable).

CHEAP because the SPINE (head + classical-super chain at fixed t1) is already
trained: every fig2 superbpe_t<t1>.merges file IS a spine. We slice it at t2 and
train only the parity tail (independent per t2 -> --parallel). t2 == total needs
no training at all (evaluates the spine itself).

Spines on disk (train_1mb): 8k explore_8k/superbpe_t2000.merges | 20k fig2_20k/
superbpe_t3000.merges | 50k fig2_50k/superbpe_t12000.merges | 80k fig2_80k/
superbpe_t20000.merges.

Run (from superparity/):
  ..\.venv\Scripts\python.exe pareto_sweep.py --tag pareto_8k --vocab 8000 ^
      --spine results\explore_8k\superbpe_t2000.merges --t1 2000 ^
      --t2 2000 2904 3872 4840 5808 6776 7744 --parallel 6
Writes results/<tag>/summary.csv + frontier.png (+ auto-refreshing frontier.html),
regenerated after every finished point. Beeps like fig2_sweep (--no-beep to mute).
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
    """Compression-vs-fairness scatter, colored by t2, with the Pareto front marked."""
    xs = [float(r["mean_tokens_per_line"]) for r in rows]
    ys = [float(r["disparity_ratio"]) for r in rows]
    fr = [float(r["t2_frac"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, c=fr, cmap="viridis", s=90, zorder=3,
                    edgecolors="white", linewidths=0.6)
    fig.colorbar(sc, ax=ax, label="t2 (fraction) — when parity starts; small = more parity")
    for x, y, f in zip(xs, ys, fr):
        ax.annotate(f"t2={f:.2f}", (x, y), fontsize=7.5,
                    xytext=(6, 4), textcoords="offset points")
    # Pareto front: minimize BOTH tokens/line and disparity
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
    ax.set_title(f"Pareto frontier @ vocab {vocab//1000}k  (t1={t1} fixed on plateau, "
                 f"sweeping t2 = parity start)", fontsize=10)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    png = out_dir / "frontier.png"
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    (out_dir / "frontier.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="{refresh_s}"><title>frontier</title></head>'
        f'<body style="font-family:system-ui;margin:24px">'
        f'<h2>Pareto frontier — vocab {vocab} (t1={t1})</h2>'
        f'<p>updated {time.strftime("%H:%M:%S")} · refreshes every {refresh_s}s</p>'
        f'<img src="frontier.png?ts={int(time.time())}" style="max-width:100%">'
        f'</body></html>', encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--data", default="train_1mb", choices=sorted(config.DATA_DIRS))
    ap.add_argument("--spine", required=True,
                    help="full superbpe .merges at this vocab (head+classical-super, fixed t1)")
    ap.add_argument("--t1", type=int, required=True,
                    help="the spine's sub->super transition (metadata + sanity checks)")
    ap.add_argument("--t2", type=int, nargs="+", required=True,
                    help="parity-start points (merge counts). t2 == total merges "
                         "= superbpe endpoint, evaluated free (no training)")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--no-beep", action="store_true")
    ap.add_argument("--langs", nargs="+", default=None,
                    help="language subset (FLORES codes); default = all 10. NOTE: a "
                         "subset changes the corpus, so use a spine TRAINED ON THE "
                         "SAME subset (fig2_sweep --langs), never a 10-language spine.")
    args = ap.parse_args()
    run.SHOW_PROGRESS = False
    run.NUM_WORKERS = 1

    langs = args.langs or config.LANGUAGES
    data_dir = config.DATA_DIRS[args.data]
    config.check_data(data_dir, langs)
    total = args.vocab - config.BYTE_ALPHABET
    inputs = config.train_files(data_dir, langs)
    devs = config.dev_files(langs)
    evals = config.eval_corpora(langs)
    if args.langs:
        print(f"LANGUAGE SUBSET ({len(langs)}): {' '.join(langs)}", flush=True)
    out = config.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    spine = run.read_merges(args.spine)
    if len(spine) < total:
        raise SystemExit(f"spine has {len(spine)} merges < total {total} needed")
    t2s = sorted(t for t in args.t2 if args.t1 <= t <= total)
    print(f"=== pareto '{args.tag}': vocab={args.vocab} (total={total}), t1={args.t1}, "
          f"data={args.data}, parallel={args.parallel} ===", flush=True)
    print(f"spine: {args.spine} ({len(spine)} merges)", flush=True)
    print(f"t2 = {t2s}  (fracs {[round(t/total, 3) for t in t2s]})", flush=True)
    print(f"parity tail lengths = {[total - t for t in t2s]}\n", flush=True)

    f = open(out / "summary.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader(); f.flush()
    lock = threading.Lock()
    rows_done: list[dict] = []
    n_done = [0]

    def train_point(t2: int):
        if t2 >= total:                       # superbpe endpoint: free evaluation
            return t2, spine[:total], 0.0
        pre = out / f"parity_t2{t2}.preload.bpe"
        run.write_merges(pre, spine[:t2], header=False)
        tail_out = out / f"parity_t2{t2}.tail.bpe"
        dt = run.run_phase(inputs, devs, total - t2, "super", "parity", tail_out, pre)
        merges = spine[:t2] + run.read_merges(tail_out)
        run.write_merges(out / f"parity_t2{t2}.merges", merges, header=True)
        return t2, merges, dt

    print(f"training {len(t2s)} points, {args.parallel} at a time ...\n", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(train_point, t) for t in t2s]
        for fut in as_completed(futures):
            t2, merges, dt = fut.result()
            row = {"recipe": "bpe_super_pa", "t1": args.t1, "t2": t2,
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
