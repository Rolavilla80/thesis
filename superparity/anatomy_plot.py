r"""ANATOMY OF ONE RUN: both metrics over the merge timeline of a single tokenizer.

The compass answers "where should t1 go" and the frontier answers "where should t2
go" -- but both compare FINISHED tokenizers. This figure shows the MECHANISM instead:
take ONE fully-trained bpe->super->pa-super run and evaluate its truncations at many
merge checkpoints (sequential determinism -> a slice IS the tokenizer mid-training).
Two stacked panels over the same x-axis (merge number):

    top    : bytes/token   -> rises the whole way; the t1 kink is visible, t2 is not
    bottom : disparity     -> flat-ish through both classical phases, PLUNGES at t2

i.e. each switch owns one metric. Eval-only (no training): minutes, not hours.

EVAL CAVEAT (deliberate): every checkpoint is scored with the SUPER pretokenizer,
including pre-t1 slices. Switching the eval pretokenizer at t1 would inject an
artificial jump exactly where we claim a real one (the known eval-pretokenizer
caveat from the scaling run). Consistent eval = any kink at t1 is merge behavior.
Consequence: pre-t1 disparity values differ from the sub-eval numbers elsewhere.

Run (from superparity/):
  ..\.venv\Scripts\python.exe -u anatomy_plot.py
  ..\.venv\Scripts\python.exe -u anatomy_plot.py --merges results\pareto_80k\parity_t259800.merges --t2 59800
"""

from __future__ import annotations

import csv
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import run
from metrics import evaluate

FIELDS = ["n_merges", "vocab", "phase",
          "mean_bytes_per_token", "mean_tokens_per_line",
          "disparity_ratio", "cv", "gini", "worst_language", "best_language"]


def _beep_finish() -> None:
    try:
        import winsound
        for f in (660, 880, 1175):
            winsound.Beep(f, 220)
    except Exception:
        pass


def checkpoints(total: int, t1: int, t2: int, step: int) -> list[int]:
    """Regular grid + dense points around the two switches + the exact switches."""
    pts = set(range(step, total, step))
    pts.update({1000, 2000, total})                       # early rise + the finish line
    for t in (t1, t2):
        pts.update({t, t + 500, t + 1000, t + 2000})      # sharpen each kink/plunge
        pts.add(max(1000, t - 1000))
    return sorted(p for p in pts if 1000 <= p <= total)


def phase_of(cut: int, t1: int, t2: int) -> str:
    if cut <= t1:
        return "bpe"
    return "classical-super" if cut <= t2 else "parity-super"


def draw(rows: list[dict], t1: int, t2: int, total: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["n_merges"] / 1000 for r in rows]
    bpt = [r["mean_bytes_per_token"] for r in rows]
    disp = [r["disparity_ratio"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25)
        ax.axvline(t1 / 1000, color="tab:green", ls="--", lw=1.2)
        ax.axvline(t2 / 1000, color="tab:red", ls="--", lw=1.2)

    ax1.plot(xs, bpt, "o-", color="tab:blue", ms=4)
    ax1.set_ylabel("bytes / token  (↑ better)")
    ax1.text(t1 / 1000, ax1.get_ylim()[1], "  t1: whitespace wall drops",
             color="tab:green", va="top", fontsize=9)
    ax1.text(t2 / 1000, ax1.get_ylim()[0], "  t2: parity objective on",
             color="tab:red", va="bottom", fontsize=9)

    ax2.plot(xs, disp, "o-", color="tab:purple", ms=4)
    ax2.set_ylabel("disparity worst/best  (↓ fairer)")
    ax2.set_xlabel("merge number (thousands)")

    fig.suptitle(f"Anatomy of one run — BPE → SuperBPE → parity-super "
                 f"(vocab {total + config.BYTE_ALPHABET:,}; t1={t1:,}, t2={t2:,})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges",
                    default=str(config.RESULTS / "pareto_80k" / "parity_t235900.merges"),
                    help="a fully-trained 3-phase .merges file (default: 80k run, "
                         "t1=20000, t2=35900 = the near-perfect-fairness point)")
    ap.add_argument("--t1", type=int, default=20000, help="sub->super switch of that run")
    ap.add_argument("--t2", type=int, default=35900,
                    help="classical->parity switch of that run")
    ap.add_argument("--step", type=int, default=4000,
                    help="checkpoint spacing in merges (denser points near t1/t2 are "
                         "added automatically)")
    ap.add_argument("--parallel", type=int, default=4,
                    help="concurrent checkpoint evals (encode releases the GIL)")
    ap.add_argument("--tag", default="anatomy", help="results/<tag>/ output folder")
    args = ap.parse_args()

    merges = run.read_merges(Path(args.merges))
    total = len(merges)
    if not (0 < args.t1 < args.t2 < total):
        raise SystemExit(f"need 0 < t1 < t2 < {total} (got t1={args.t1}, t2={args.t2})")
    evals = config.eval_corpora()
    out = config.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    cps = checkpoints(total, args.t1, args.t2, args.step)
    print(f"=== anatomy: {args.merges} ===", flush=True)
    print(f"{total} merges (vocab {total + 256:,}), t1={args.t1}, t2={args.t2}; "
          f"{len(cps)} checkpoints, parallel={args.parallel}\n", flush=True)

    lock = threading.Lock()
    done = [0]
    rows: dict[int, dict] = {}
    t0 = time.perf_counter()

    def eval_cut(cut: int) -> tuple[int, dict]:
        row = {"n_merges": cut, "vocab": cut + 256, "phase": phase_of(cut, args.t1, args.t2)}
        row.update(evaluate(merges[:cut], "super", evals))
        return cut, row

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(eval_cut, c) for c in cps]
        for fut in as_completed(futures):
            cut, row = fut.result()
            with lock:
                rows[cut] = row
                done[0] += 1
                print(f"  [{done[0]:2d}/{len(cps)}] merges={cut:6d} "
                      f"({row['phase']:15s})  bytes/tok={row['mean_bytes_per_token']:.4f}  "
                      f"disp={row['disparity_ratio']:.3f}", flush=True)

    ordered = [rows[c] for c in sorted(rows)]
    with open(out / "anatomy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ordered)

    png = out / "anatomy.png"
    draw(ordered, args.t1, args.t2, total, png)
    print(f"\nDONE in {(time.perf_counter()-t0)/60:.1f} min -> {png}", flush=True)
    _beep_finish()


if __name__ == "__main__":
    main()
